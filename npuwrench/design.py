# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
# Derived from amd/IRON iron/operators/flm/lm_head/design.py: RMS norm, softcap and RTP removed;
# two resident stages on disjoint columns; stage A size-agnostic (one round per activation).
"""Two resident q4nx GEMV stages, 4 columns each, serving every linear layer of Hy-Memory .

Stage A (cols 0-3, dim 2048) is size-agnostic: each core handles ONE round per activation it
acquires, and the instruction stream re-sends the activation every round, so the output size
(qkv 5120, gate|up 1536, o-proj halves 2048, LM-head blocks 5120) lives only in the instruction
stream. Stage D (cols 4-7, dim 768) is the MoE down with the SwiGLU prologue (n=2048).
"""
import numpy as np
from ml_dtypes import bfloat16
from aie.helpers.npdtypes import np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from iron.operators.flm.q4nx import K_TILE, M_TILE, packed_bytes

DEPTH = 2
STACK_SIZE = 10 * 1024
bf16 = np.dtype[bfloat16]
COLS = 4


def _set(dev, tag, dim, col0, kern, swiglu, rounds_per_x, wdepth=2):
    """rounds_per_x: rounds a core runs per acquired activation (1 = size-agnostic)."""
    ROWS = len(dev.core_rows)
    w_blk_ty, x_ty, y_acc_ty, sums_ty, _ = kern.k_block.arg_types()
    y_blk_ty, _ = kern.k_epi.arg_types()
    (w_blk,) = np_ndarray_type_get_shape(w_blk_ty)
    IT = [Tile(col0 + j, 0) for j in range(COLS)]

    def core_fn(x_in, w_in, y_out, k_sums, k_zero, k_block, k_epi, y_acc, sums, hbuf):
        x = x_in.acquire(1)
        if swiglu:
            k_sums(x, hbuf, sums)
            x = hbuf
        else:
            k_sums(x, sums)
        for _ in range_(rounds_per_x):
            k_zero(y_acc)
            for k in range_(dim // K_TILE):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    xin = 2 * dim if swiglu else dim
    of_x = ObjectFifo(np.ndarray[(xin,), bf16], name=f"{tag}x")
    x_prod = of_x.prod(tile=IT[0])
    workers, w_prods, y_conses = [], [], []
    for j in range(COLS):
        of_w = ObjectFifo(np.ndarray[(ROWS, w_blk), bf16], depth=wdepth, name=f"{tag}w{j}")
        w_prods.append(of_w.prod(tile=IT[j]))
        w_cores = of_w.cons().split([w_blk * i for i in range(ROWS)], obj_types=[w_blk_ty] * ROWS,
                                    names=[f"{tag}w{j}_{i}" for i in range(ROWS)])
        of_y = ObjectFifo(np.ndarray[(ROWS, M_TILE), bf16], name=f"{tag}y{j}")
        y_conses.append(of_y.cons(tile=IT[j]))
        y_cores = of_y.prod().join([M_TILE * i for i in range(ROWS)], obj_types=[y_blk_ty] * ROWS,
                                   names=[f"{tag}y{j}_{i}" for i in range(ROWS)])
        for i, row in enumerate(dev.core_rows):
            workers.append(Worker(
                core_fn,
                [of_x.cons(), w_cores[i].cons(), y_cores[i].prod(), kern.k_swiglu if swiglu else kern.k_sums,
                 kern.k_zero, kern.k_block, kern.k_epi, Buffer(type=y_acc_ty, name=f"{tag}yacc_{i}_{j}"),
                 Buffer(type=sums_ty, name=f"{tag}sums_{i}_{j}"),
                 Buffer(type=x_ty if swiglu else np.ndarray[(16,), bf16], name=f"{tag}h_{i}_{j}")],
                tile=Tile(col0 + j, row), stack_size=STACK_SIZE))
    return dict(workers=workers, eps=[x_prod, w_prods, y_conses], dim=dim, xin=xin, rpx=rounds_per_x)


def uni_gemv(dev, which, n_out, wdepth=2, win=2, *, kerns):
    """which: "A" (dim 2048, any n_out multiple of 512) or "D" (MoE down, n_out 2048)."""
    ROWS = len(dev.core_rows); per_round = COLS * ROWS * M_TILE
    A = _set(dev, "A", 2048, 0, kerns[(2048, False)], False, 1, wdepth)
    Dd = _set(dev, "D", 768, COLS, kerns[(768, True)], True, 2048 // per_round, wdepth)
    S = {"A": A, "D": Dd}[which]
    rounds = n_out // per_round
    assert rounds * per_round == n_out and (which == "A" or n_out == 2048)
    dim = S["dim"]
    M_PER_COL = ROWS * M_TILE; W_PER_COL = packed_bytes(M_PER_COL * dim) // 4
    taps = lambda n, e: [TensorAccessPattern((n, e), i * e, [1, 1, 1, e], [0, 0, 0, 1]) for i in range(n)]
    y_taps, w_taps = taps(rounds * COLS, M_PER_COL), taps(rounds * COLS, W_PER_COL)
    y_ty = np.ndarray[(n_out,), bf16]
    w_ty = np.ndarray[(packed_bytes(n_out * dim) // 4,), np.dtype[np.uint32]]
    x_ty = np.ndarray[(S["xin"],), bf16]

    def sequence(Y, W, X, ax, aw, ay, dx, dw, dy):
        x_prod, w_prods, y_conses = (ax, aw, ay) if which == "A" else (dx, dw, dy)
        window = []
        for rnd in range(rounds):
            tg = TaskGroup()
            if rnd % S["rpx"] == 0:  # freed with this round's group (the shim has 16 BDs)
                x_prod.fill(X, wait=False, group=tg)
            for c in range(COLS):
                i = rnd * COLS + c
                y_conses[c].drain(Y, y_taps[i], wait=True, group=tg)
                w_prods[c].fill(W, w_taps[i], wait=False, group=tg)
            window.append(tg)
            if len(window) > win:
                window.pop(0).finish()
        for tg in window:
            tg.finish()

    rt = Runtime(sequence, [y_ty, w_ty, x_ty] + A["eps"] + Dd["eps"])
    return Program(dev, rt, workers=A["workers"] + Dd["workers"]).resolve_program()


def uni_multi(dev, which, n_out, E, wdepth=2, win=2, *, kerns):
    """E experts of one stage in one dispatch (E <= 3: a dispatch carries about 5 buffer arguments).
    Args: Y (E x n_out), X (A: one activation; D: E x 1536 gate|up), W0..W{E-1}. Same xclbin as uni_gemv."""
    ROWS = len(dev.core_rows); per_round = COLS * ROWS * M_TILE
    A = _set(dev, "A", 2048, 0, kerns[(2048, False)], False, 1, wdepth)
    Dd = _set(dev, "D", 768, COLS, kerns[(768, True)], True, 2048 // per_round, wdepth)
    S = {"A": A, "D": Dd}[which]
    rounds = n_out // per_round; dim = S["dim"]; xin = S["xin"]; per_x = which == "D"
    M_PER_COL = ROWS * M_TILE; W_PER_COL = packed_bytes(M_PER_COL * dim) // 4
    wt = [TensorAccessPattern((rounds * COLS, W_PER_COL), i * W_PER_COL, [1, 1, 1, W_PER_COL], [0, 0, 0, 1])
          for i in range(rounds * COLS)]
    yt = [[TensorAccessPattern((E * n_out,), e * n_out + i * M_PER_COL, [1, 1, 1, M_PER_COL], [0, 0, 0, 1])
           for i in range(rounds * COLS)] for e in range(E)]
    xt = [TensorAccessPattern(((E if per_x else 1) * xin,), (e * xin if per_x else 0), [1, 1, 1, xin], [0, 0, 0, 1])
          for e in range(E)]
    y_ty = np.ndarray[(E * n_out,), bf16]
    x_ty = np.ndarray[((E if per_x else 1) * xin,), bf16]
    w_ty = np.ndarray[(packed_bytes(n_out * dim) // 4,), np.dtype[np.uint32]]

    def sequence(Y, X, *rest):
        Ws = rest[:E]; ax, aw, ay, dx, dw, dy = rest[E:]
        x_prod, w_prods, y_conses = (ax, aw, ay) if which == "A" else (dx, dw, dy)
        window = []
        for e in range(E):
            for rnd in range(rounds):
                tg = TaskGroup()
                if rnd % S["rpx"] == 0:
                    x_prod.fill(X, xt[e], wait=False, group=tg)
                for c in range(COLS):
                    i = rnd * COLS + c
                    y_conses[c].drain(Y, yt[e][i], wait=True, group=tg)
                    w_prods[c].fill(Ws[e], wt[i], wait=False, group=tg)
                window.append(tg)
                if len(window) > win:
                    window.pop(0).finish()
        for tg in window:
            tg.finish()

    rt = Runtime(sequence, [y_ty, x_ty] + [w_ty] * E + A["eps"] + Dd["eps"])
    return Program(dev, rt, workers=A["workers"] + Dd["workers"]).resolve_program()
