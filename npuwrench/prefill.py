# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Batched prompt processing (prefill) on the NPU for HyMemoryNPU.

The whole prompt goes through each layer at once. Weights are the same resident q4nx buffers decode
streams: IRON's DequantBFP, reading them in place (prefill_ops.DequantLayout), writes the bfp16 B that
IRON's flm.GEMM multiplies. Two NPU programs (dequant, GEMM; shapes live in the instruction streams).

Per layer:  dequant qkv, o -> GEMM qkv -> CPU: q/k norm, RoPE, causal attention, KV cache
            -> GEMM o -> CPU: residual, norm, router, group rows by expert
            -> dequant the active experts' gate|up and down -> GEMM gate|up -> CPU: SiLU(gate)*up
            -> GEMM down (K padded 768 -> 1024 with zeros) -> CPU: weighted scatter-add
The LM head runs once, on the last token, through the decode path.
"""
import time

import numpy as np
from ml_dtypes import bfloat16

from .engine import D, HD, NE, NH, NKV, TOPK, _bf, _rms

ROW_TILE = 256  # flm.GEMM: M must be a multiple of 64 x 4 compute rows


def _pad(n, m=ROW_TILE):
    return (n + m - 1) // m * m


def _bucket(n):
    """Expert row counts round up to a power of two (>= 256): fewer GEMM instruction streams (HRX caches 16)."""
    b = ROW_TILE
    while b < n:
        b *= 2
    return b


class Prefill:
    def __init__(self, eng, max_tokens=4096, verbose=True):
        from aie.utils.hostruntime.hrxruntime.tensor import HRXTensor
        from aie.utils.npukernel import NPUKernel
        from iron.operators.flm import GEMM
        from .prefill_ops import DequantLayout
        self.e, self.T, self.NPUKernel, self.GEMM = eng, HRXTensor, NPUKernel, GEMM
        self.rt = eng.rt
        self.Tmax = _bucket(max_tokens)
        t0 = time.time()
        shapes = {"qkv": (2048, 5120, 8), "o": (2048, 2048, 8), "gu": (2048, 1536, 8), "d": (1024, 2048, 3)}
        self.dq = {k: DequantLayout(K=K, N=N, src_kb=kb) for k, (K, N, kb) in shapes.items()}
        for o in self.dq.values():
            o.compile()
        x0 = self.dq["qkv"].xclbin_artifact.filename
        self.hd = {k: self.rt.load(NPUKernel(x0, o.insts_artifact.filename)) for k, o in self.dq.items()}
        self.gemm, self.hg, self.gx = {}, {}, None
        B = lambda n: HRXTensor((n,), dtype=np.uint8)
        self.Bqkv, self.Bo = B(self.dq["qkv"].packed_size()), [B(self.dq["o"].packed_size()) for _ in range(2)]
        self.Bgu = [B(self.dq["gu"].packed_size()) for _ in range(NE + 1)]
        self.Bd = [B(self.dq["d"].packed_size()) for _ in range(NE + 1)]
        bfT = lambda *s: HRXTensor(s, dtype=bfloat16)
        Tm = self.Tmax
        self.Ax, self.Cqkv = bfT(Tm, 2048), bfT(Tm, 5120)
        self.Ao, self.Co = [bfT(Tm, 2048) for _ in range(2)], [bfT(Tm, 2048) for _ in range(2)]
        # one slot per expert (128 routed + shared): rows <= Tmax
        self.Ae = [bfT(Tm, 2048) for _ in range(NE + 1)]
        self.Ce = [bfT(Tm, 1536) for _ in range(NE + 1)]
        self.He = [bfT(Tm, 1024) for _ in range(NE + 1)]
        self.De = [bfT(Tm, 2048) for _ in range(NE + 1)]
        for h in self.He:
            h.numpy()[:] = 0  # columns 768..1023 stay zero: the K padding of the down GEMM
            h._sync_to_device()
        # HRX flushes (or invalidates) a whole buffer around every dispatch. The bfp16 B scratch is written and
        # read only by the NPU, so it needs neither; the row buffers are flushed/invalidated over the rows used.
        for t in [self.Bqkv] + self.Bo + self.Bgu + self.Bd:
            t._sync_to_device = t._sync_from_device = lambda: None
        for t in [self.Ax, self.Cqkv] + self.Ao + self.Co + self.Ae + self.Ce + self.He + self.De:
            t._sync_to_device = lambda: None
        self.stats = {"npu_s": 0.0, "cpu_s": 0.0, "compile_s": 0.0}
        self._tri = np.triu(np.full((256, 256), -np.inf, np.float32), 1)
        if verbose:
            print(f"prefill ready in {time.time() - t0:.1f}s (max {self.Tmax} tokens)", flush=True)

    def _g(self, M, K, N):
        """flm.GEMM handle for one shape (one xclbin for all shapes; one instruction stream each)."""
        key = (M, K, N)
        if key not in self.hg:
            s = time.time()
            op = self.GEMM(M=M, K=K, N=N)
            op.compile()
            if self.gx is None:
                self.gx = op.xclbin_artifact.filename
            self.hg[key] = self.rt.load(self.NPUKernel(self.gx, op.insts_artifact.filename))
            self.stats["compile_s"] += time.time() - s
        return self.hg[key]

    @staticmethod
    def _wr(t, n, value, cols=None):
        """Write rows [0, n) (optionally columns [0, cols)) and flush just those rows to the device."""
        v = t.data
        if cols is None:
            v[:n] = value
        else:
            v[:n, :cols] = value
        t._ctx.flush_range(t._buf, 0, n * v[0].nbytes)

    @staticmethod
    def _rd(t, n):
        """Invalidate and read rows [0, n) written by the NPU."""
        v = t.data
        t._ctx.invalidate_range(t._buf, 0, n * v[0].nbytes)
        return v[:n]

    def _run(self, chain):
        s = time.perf_counter()
        self.rt.run_chain(chain)
        self.stats["npu_s"] += time.perf_counter() - s

    def _attention(self, l, q, k, v, T):
        """Causal GQA over positions 0..T-1 with batched BLAS matmuls, in query blocks so only the
        lower triangle (plus one block diagonal) is computed."""
        e = self.e
        G = NH // NKV
        e.Kt[l, :, :, :T] = k.transpose(1, 2, 0)                 # (NKV, HD, T)
        e.Vh[l, :, :T] = v.transpose(1, 0, 2)                    # (NKV, T, HD)
        Kt, Vh = e.Kt[l], e.Vh[l]
        qh = (q * np.float32(1.0 / np.sqrt(HD))).reshape(T, NKV, G, HD).transpose(1, 2, 0, 3)  # scale folded into q
        out = np.empty((NKV, G, T, HD), np.float32)
        for s0 in range(0, T, 256):
            s1 = min(T, s0 + 256)
            tb = s1 - s0
            qb = np.ascontiguousarray(qh[:, :, s0:s1]).reshape(NKV, G * tb, HD)
            sc = np.matmul(qb, Kt[:, :, :s1]).reshape(NKV, G, tb, s1)  # (NKV, G, Tb, s1)
            # keys before the block are all visible; only the diagonal block is causal
            sc[..., s0:s1] += self._tri[:tb, :tb]
            sc -= sc.max(-1, keepdims=True)
            np.exp(sc, out=sc)
            den = sc.sum(-1, keepdims=True)                              # normalize the output, not P
            o = np.matmul(sc.reshape(NKV, G * tb, s1), Vh[:, :s1]).reshape(NKV, G, tb, HD)
            out[:, :, s0:s1] = o / den
        return out.transpose(2, 0, 1, 3).reshape(T, NH * HD)

    def _experts(self, l, xn, groups, T):
        """groups: list of (slot, weight-index, row indices, per-row weights or None). Returns sum (T, 2048)."""
        e = self.e
        ch = []
        for slot, wi, rows, _ in groups:
            ch.append((self.hd["gu"], [e.Wgu[l][wi], self.Bgu[slot]]))
            ch.append((self.hd["d"], [e.Wd[l][wi], self.Bd[slot]]))
        self._run(ch)
        s = time.perf_counter()
        ch = []
        for slot, wi, rows, _ in groups:
            M = _bucket(len(rows))
            self._wr(self.Ae[slot], len(rows), _bf(xn[rows]))
            ch.append((self._g(M, 2048, 1536), [self.Ae[slot], self.Bgu[slot], self.Ce[slot]]))
        self.stats["cpu_s"] += time.perf_counter() - s
        self._run(ch)
        s = time.perf_counter()
        ch = []
        for slot, wi, rows, _ in groups:
            n, M = len(rows), _bucket(len(rows))
            c = self._rd(self.Ce[slot], n).astype(np.float32)
            g = c[:, :768]
            self._wr(self.He[slot], n, _bf(g / (1.0 + np.exp(-g)) * c[:, 768:]), cols=768)
            ch.append((self._g(M, 1024, 2048), [self.He[slot], self.Bd[slot], self.De[slot]]))
        self.stats["cpu_s"] += time.perf_counter() - s
        self._run(ch)
        s = time.perf_counter()
        out = np.zeros((T, D), np.float32)
        for slot, wi, rows, w in groups:
            y = self._rd(self.De[slot], len(rows)).astype(np.float32)
            if w is None:
                out[rows] += y
            else:
                out[rows] += w[:, None] * y  # a token picks an expert at most once
        self.stats["cpu_s"] += time.perf_counter() - s
        return out

    def run(self, ids):
        """Fill the KV cache for ``ids`` (positions 0..T-1); returns the last token's logits."""
        e, T = self.e, len(ids)
        assert T <= self.Tmax
        Tp = _pad(T)
        x = np.array(e.embed[ids], np.float32)
        pos = np.arange(T)
        cos, sin = e.cos[pos][:, None, :], e.sin[pos][:, None, :]

        def rope(t):
            h = HD // 2
            return np.concatenate([t[..., :h] * cos - t[..., h:] * sin, t[..., h:] * cos + t[..., :h] * sin], -1)

        for l in range(e.NL):
            L = e.L[l]
            s = time.perf_counter()
            self._wr(self.Ax, T, _bf(_rms(x, L["input_layernorm"])))
            self.stats["cpu_s"] += time.perf_counter() - s
            self._run([(self.hd["qkv"], [e.Wqkv[l], self.Bqkv]),
                       (self.hd["o"], [e.Wo[l][0], self.Bo[0]]), (self.hd["o"], [e.Wo[l][1], self.Bo[1]])])
            self._run([(self._g(Tp, 2048, 5120), [self.Ax, self.Bqkv, self.Cqkv])])
            s = time.perf_counter()
            qkv = self._rd(self.Cqkv, T).astype(np.float32)
            q = rope(_rms(qkv[:, :NH * HD].reshape(T, NH, HD), L["q_norm"]))
            k = rope(_rms(qkv[:, NH * HD:(NH + NKV) * HD].reshape(T, NKV, HD), L["k_norm"]))
            v = qkv[:, (NH + NKV) * HD:].reshape(T, NKV, HD)
            attn = self._attention(l, q, k, v, T)
            self._wr(self.Ao[0], T, _bf(attn[:, :2048]))
            self._wr(self.Ao[1], T, _bf(attn[:, 2048:]))
            self.stats["cpu_s"] += time.perf_counter() - s
            g = self._g(Tp, 2048, 2048)
            self._run([(g, [self.Ao[0], self.Bo[0], self.Co[0]]), (g, [self.Ao[1], self.Bo[1], self.Co[1]])])
            s = time.perf_counter()
            x = x + self._rd(self.Co[0], T).astype(np.float32) + self._rd(self.Co[1], T).astype(np.float32)
            xn = _rms(x, L["post_attention_layernorm"])
            if l == 0:  # dense FFN: 9 width-768 chunks, every row, weight 1
                groups = [(j, j, pos, None) for j in range(9)]
            else:
                p = 1.0 / (1.0 + np.exp(-(xn @ L["router"].T)))                 # (T, 128)
                ex = np.argsort(-(p + L["bias"]), axis=1)[:, :TOPK]              # bias picks experts only
                w = np.take_along_axis(p, ex, 1)
                w = w / np.maximum(w.sum(1, keepdims=True), 6.103515625e-5) * e.scale
                groups = []
                for ei in range(NE):
                    r, c = np.nonzero(ex == ei)
                    if len(r):
                        groups.append((ei, ei, r, w[r, c].astype(np.float32)))
                groups.append((NE, NE, pos, None))                               # shared expert
            self.stats["cpu_s"] += time.perf_counter() - s
            x = x + self._experts(l, xn, groups, T)
        e.stats["steps"] += T
        np.copyto(e.bx.numpy(), _bf(_rms(x[-1], e.norm)))
        e._run([(e.h["q"], [e.blm[j], e.Wlm[j], e.bx]) for j in range(24)])
        return np.concatenate([b.numpy().astype(np.float32) for b in e.blm])[:e.vocab]
