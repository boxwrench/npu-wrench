# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""The q4nx block order ExpertGEMV reads, its inverse, and the encoder.

The lm_head design fixes the order: out-feature ``n`` is
``((round * cols + col) * rows + row) * 32 + m`` and its k-th block sits at
``((round * cols + col) * k_blocks + k) * rows + row``. A column therefore reads
one contiguous run of ``k_blocks * rows`` blocks per round, and the rows pull
one block each. The block bytes are the standard q4nx block (32 out x 256 in,
group 32, bf16 scale then min, adjacent-code nibbles).
"""

import numpy as np

from iron.operators.flm.dequant.reference import _block_origins
from iron.operators.flm.lm_head.reference import dequantize_block
from iron.operators.flm.q4nx import BLOCK_BYTES, GROUP, K_TILE, M_TILE

GROUP_N = M_TILE  # out-features per block


def dest_index(n0: int, k0: int, dim: int, cols: int, rows: int) -> int:
    """Blob block index of the block whose first weight is (n0, k0)."""
    k_blocks = dim // K_TILE
    slice_ = n0 // (rows * GROUP_N)
    rnd, col = divmod(slice_, cols)
    row = (n0 % (rows * GROUP_N)) // GROUP_N
    k = k0 // K_TILE
    return ((rnd * cols + col) * k_blocks + k) * rows + row


def dest_origin(index: int, dim: int, cols: int, rows: int):
    """(n0, k0) of blob block ``index``."""
    k_blocks = dim // K_TILE
    q, row = divmod(index, rows)
    slice_, k = divmod(q, k_blocks)
    rnd, col = divmod(slice_, cols)
    return (rnd * cols + col) * rows * GROUP_N + row * GROUP_N, k * K_TILE


def block_origins_layout(dim, n_out, cols, rows):
    """(n0, k0) per blob block, in blob order."""
    per_round = cols * rows * GROUP_N
    return [
        dest_origin(i, dim, cols, rows)
        for i in range(n_out // per_round * cols * (dim // K_TILE) * rows)
    ]


def dequantize_layout(qw, dim, n_out, cols, rows):
    """A blob in this layout to (n_out, dim) float64 weights."""
    b = np.asarray(qw).view(np.uint8).reshape(-1, BLOCK_BYTES)
    out = np.empty((n_out, dim), np.float64)
    for i, (n0, k0) in enumerate(block_origins_layout(dim, n_out, cols, rows)):
        out[n0 : n0 + GROUP_N, k0 : k0 + K_TILE] = dequantize_block(b[i])
    return out


def _bf16_bits(x):
    """Round to nearest-even bf16, as hy_convert.py does."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16)


def encode_block(wb):
    """A (K_TILE, GROUP_N) float weight tile to one q4nx block."""
    wg = np.asarray(wb, np.float32).reshape(K_TILE // GROUP, GROUP, GROUP_N)
    mn = wg.min(axis=1)
    mx = wg.max(axis=1)
    sc = (mx - mn) / 15.0
    sc = np.where(sc == 0, 1.0, sc)
    q = np.rint((wg - mn[:, None, :]) / sc[:, None, :]).clip(0, 15).astype(np.uint8)
    qs = np.empty((2, K_TILE, 8), dtype=np.uint8)
    for p in range(2):
        for j in range(8):
            lo = q[:, :, p * 16 + 2 * j]
            hi = q[:, :, p * 16 + 2 * j + 1]
            qs[p, :, j] = (lo | (hi << 4)).reshape(-1)
    return np.concatenate(
        [
            _bf16_bits(sc).view(np.uint8).reshape(-1),
            _bf16_bits(mn).view(np.uint8).reshape(-1),
            qs.reshape(-1),
        ]
    )


def encode_layout(W, dim, n_out, cols, rows):
    """A (dim, n_out) float matrix to a blob in this layout.

    ``W`` is in-feature major, the GGUF dequantize order.
    """
    W = np.asarray(W, np.float32)
    out = np.empty(
        (n_out // (cols * rows * GROUP_N) * cols * (dim // K_TILE) * rows, BLOCK_BYTES),
        np.uint8,
    )
    for i, (n0, k0) in enumerate(block_origins_layout(dim, n_out, cols, rows)):
        out[i] = encode_block(W[k0 : k0 + K_TILE, n0 : n0 + GROUP_N])
    return out.reshape(-1)


def repack_from_dequant_order(blob, dim, n_out, rows, cols):
    """Permute a ``flm.DequantBFP`` blob (dequant order) into this layout.

    The block bytes are identical; only the block order differs. ``dim`` must
    be a multiple of 512 for ``_block_origins`` to cover it.
    """
    if dim % 512:
        raise ValueError("repack needs dim % 512 == 0")
    b = np.asarray(blob).view(np.uint8).reshape(-1, BLOCK_BYTES)
    n_blocks = (dim // K_TILE) * (n_out // GROUP_N)
    out = np.empty_like(b)
    for i, (n0, k0) in enumerate(_block_origins(n_blocks, dim)):
        out[dest_index(n0, k0, dim, cols, rows)] = b[i]
    return out.reshape(-1)
