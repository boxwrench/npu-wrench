# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""MSE-searched q4nx encoder: same block bytes as layout.encode_block, better scale/min.

Per (32-k group, out-feature): try scale candidates around (max-min)/15, quantize, refit
(scale, min) by least squares on the codes, keep the lowest squared error (optionally
weighted per input column by an activation importance vector, like llama.cpp imatrix).
"""
import numpy as np
from . import layout
from .layout import K_TILE, GROUP, GROUP_N, _bf16_bits

IMP = None  # optional (dim,) importance; set by caller; sliced per block via encode_layout hook


def _bf16_round(x):
    return (_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def fit(wg, imp=None, steps=np.linspace(-0.6, 0.6, 13)):
    """wg (G, 32, N) -> codes, scale, min (G, N); weighted LSQ refit."""
    w = np.ones_like(wg) if imp is None else np.broadcast_to(imp, wg.shape)
    mn0, mx0 = wg.min(1), wg.max(1)
    rng = np.maximum(mx0 - mn0, 1e-12)
    best = None
    for st in steps:
        sc = rng / (15.0 + st)
        mn = mn0.copy()
        for _ in range(2):
            q = np.rint((wg - mn[:, None]) / sc[:, None]).clip(0, 15)
            sw, sq, sqq = w.sum(1), (w * q).sum(1), (w * q * q).sum(1)
            sv, sqv = (w * wg).sum(1), (w * q * wg).sum(1)
            det = sw * sqq - sq * sq
            ok = det > 1e-20
            sc = np.where(ok, (sw * sqv - sq * sv) / np.where(ok, det, 1), sc)
            mn = np.where(ok, (sqq * sv - sq * sqv) / np.where(ok, det, 1), mn)
            sc = np.where(sc <= 0, rng / 15.0, sc)
        scb, mnb = _bf16_round(sc), _bf16_round(mn)
        q = np.rint((wg - mnb[:, None]) / scb[:, None]).clip(0, 15)
        err = (w * (mnb[:, None] + scb[:, None] * q - wg) ** 2).sum(1)
        if best is None:
            best = [err, q, scb, mnb]
        else:
            b = err < best[0]
            best[0] = np.where(b, err, best[0])
            best[1] = np.where(b[:, None], q, best[1])
            best[2] = np.where(b, scb, best[2]); best[3] = np.where(b, mnb, best[3])
    return best[1].astype(np.uint8), best[2], best[3]


def encode_block(wb, imp=None):
    wg = np.asarray(wb, np.float32).reshape(K_TILE // GROUP, GROUP, GROUP_N)
    ib = None if imp is None else np.asarray(imp, np.float32).reshape(K_TILE // GROUP, GROUP, 1)
    q, sc, mn = fit(wg, ib)
    qs = np.empty((2, K_TILE, 8), dtype=np.uint8)
    for p in range(2):
        for j in range(8):
            qs[p, :, j] = (q[:, :, p * 16 + 2 * j] | (q[:, :, p * 16 + 2 * j + 1] << 4)).reshape(-1)
    return np.concatenate([_bf16_bits(sc).view(np.uint8).reshape(-1),
                           _bf16_bits(mn).view(np.uint8).reshape(-1), qs.reshape(-1)])


def encode_layout(W, dim, n_out, cols, rows, imp=None):
    """(dim, n_out) -> blob, like layout.encode_layout; imp is (dim,) or None."""
    W = np.asarray(W, np.float32)
    origins = list(layout.block_origins_layout(dim, n_out, cols, rows))
    out = np.empty((len(origins), layout.BLOCK_BYTES), np.uint8)
    for i, (n0, k0) in enumerate(origins):
        out[i] = encode_block(W[k0:k0 + K_TILE, n0:n0 + GROUP_N],
                              None if imp is None else imp[k0:k0 + K_TILE])
    return out.reshape(-1)


def encode_layout_fast(W, dim, n_out, cols, rows, imp=None):
    """encode_layout with one whole-matrix fit (groups are 32 consecutive k per out-feature)."""
    W = np.asarray(W, np.float32)                     # (dim, n_out)
    wg = W.reshape(dim // GROUP, GROUP, n_out)
    ib = None if imp is None else np.asarray(imp, np.float32).reshape(dim // GROUP, GROUP, 1)
    q, sc, mn = fit(wg, ib)                            # q (dim/32, 32, n), sc/mn (dim/32, n)
    origins = list(layout.block_origins_layout(dim, n_out, cols, rows))
    out = np.empty((len(origins), layout.BLOCK_BYTES), np.uint8)
    g = K_TILE // GROUP
    for i, (n0, k0) in enumerate(origins):
        gs = slice(k0 // GROUP, k0 // GROUP + g)
        qb = q[gs, :, n0:n0 + GROUP_N]
        qs = np.empty((2, K_TILE, 8), dtype=np.uint8)
        for p in range(2):
            for j in range(8):
                qs[p, :, j] = (qb[:, :, p * 16 + 2 * j] | (qb[:, :, p * 16 + 2 * j + 1] << 4)).reshape(-1)
        out[i] = np.concatenate([_bf16_bits(sc[gs, n0:n0 + GROUP_N]).view(np.uint8).reshape(-1),
                                 _bf16_bits(mn[gs, n0:n0 + GROUP_N]).view(np.uint8).reshape(-1),
                                 qs.reshape(-1)])
    return out.reshape(-1)
