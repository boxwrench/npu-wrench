# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Hy-Memory-30B-A3B hybrid engine for AMD Strix Halo / Strix Point (XDNA2 NPU).

Every linear layer runs on the NPU as a q4nx matrix-vector product inside ONE resident program
(two 4-column stages, see design.py), so no xclbin switch happens during a token:

  stage A (dim 2048): qkv 2048->5120, o-proj as two 2048-wide halves, MoE gate|up 2048->1536,
                      layer-0 dense FFN as 9 width-768 chunks, LM head as 24 blocks of 5120
  stage D (dim 768) : MoE down 768->2048 with SiLU(gate)*up computed on the cores

RMS norms, RoPE, attention over the KV cache and the sigmoid router run on the CPU in numpy.
Decode only: a prompt is fed one token at a time (an NPU prefill path is future work).
"""
import json
import os
import time

import numpy as np
from ml_dtypes import bfloat16

from . import st

D, NH, NKV, HD, NE, TOPK = 2048, 32, 4, 128, 128, 8
EPS = 1e-5


def _bf(x):
    return np.ascontiguousarray(x, np.float32).astype(bfloat16)


def _rms(x, w):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w


class HyMemoryNPU:
    def __init__(self, weights, max_ctx=4096, experts_per_dispatch=1, prefill_tokens=0, verbose=True):
        os.environ.setdefault("NPU_RUNTIME", "hrx")
        import aie.utils as aie_utils
        from aie.iron.device import NPU2
        aie_utils.set_current_device(NPU2())
        from aie.utils.hostruntime.hrxruntime.hostruntime import HRXHostRuntime
        from aie.utils.hostruntime.hrxruntime.tensor import HRXTensor
        from aie.utils.npukernel import NPUKernel
        from .op import UniGEMV

        self.cfg = json.load(open(os.path.join(weights, "config.json")))
        c = self.cfg
        self.NL, self.vocab, self.scale = c["num_hidden_layers"], c["vocab_size"], c["router_scaling_factor"]
        theta = c["rope_theta"]
        self.E = experts_per_dispatch
        self.skip_weight_flush = os.environ.get("NPUWRENCH_WEIGHT_FLUSH", "0") != "1"
        assert self.E in (1, 3)
        def T(a):
            # Weights are written once, here (the constructor flushes them to the device). HRX flushes every
            # argument's cache lines before each dispatch; for never-modified weights that is pure overhead.
            t = HRXTensor(np.ascontiguousarray(a), dtype=np.uint8)
            if self.skip_weight_flush:
                t._sync_to_device = lambda: None
            return t
        # down blobs get a 4-block zero tail: prefill dequantizes them with K padded 768 -> 1024
        tail = np.zeros(4 * 5120, np.uint8)
        Td = lambda a: T(np.concatenate([np.asarray(a), tail]))
        B = lambda n: HRXTensor((n,), dtype=bfloat16)
        self.rt = HRXHostRuntime()
        t0 = time.time()
        specs = {"q": ("A", 5120, 1), "g": ("A", 1536, 1), "o": ("A", 2048, 1), "d": ("D", 2048, 1)}
        if self.E == 3:
            specs.update(g3=("A", 1536, 3), d3=("D", 2048, 3))
        ops = {k: UniGEMV(w, n, e) for k, (w, n, e) in specs.items()}
        for o in ops.values():
            o.compile()
        x0 = ops["q"].xclbin_artifact.filename  # every instruction stream runs on this one program
        self.h = {k: self.rt.load(NPUKernel(x0, o.insts_artifact.filename)) for k, o in ops.items()}
        t1 = time.time()

        self.L = []
        self.Wqkv, self.Wo, self.Wgu, self.Wd = [], [], [], []
        for l in range(self.NL):
            r = st.Reader(os.path.join(weights, f"layer_{l:02d}.safetensors"))
            d = {k: np.array(r.get(k)) for k in ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")}
            self.Wqkv.append(T(r.get("qkv")))
            self.Wo.append((T(r.get("o0")), T(r.get("o1"))))
            if l == 0:  # dense FFN as 9 chunks of the MoE stages, weight 1, no shared expert
                self.Wgu.append([T(r.get(f"C{j}_gu")) for j in range(9)])
                self.Wd.append([Td(r.get(f"C{j}_d")) for j in range(9)])
            else:
                d["router"] = np.array(r.get("router")); d["bias"] = np.array(r.get("bias"))
                self.Wgu.append([T(r.get(f"E{e}_gu")) for e in range(NE)] + [T(r.get("Es_gu"))])
                self.Wd.append([Td(r.get(f"E{e}_d")) for e in range(NE)] + [Td(r.get("Es_d"))])
            self.L.append(d)
        r = st.Reader(os.path.join(weights, "head.safetensors"))
        self.Wlm = [T(r.get(f"lm_{j}")) for j in range(24)]
        self.embed = r.get("embed")          # (vocab, 2048) float32 view of BF16
        self.norm = np.array(r.get("norm"))
        if verbose:
            print(f"compiled in {t1 - t0:.1f}s, weights resident in {time.time() - t1:.1f}s", flush=True)

        self.bx, self.bqkv = B(D), B(5120)
        self.ba, self.bo = [B(2048), B(2048)], [B(D), B(D)]
        self.bgu, self.bd = [B(1536) for _ in range(9)], [B(D) for _ in range(9)]
        self.bgu3, self.bd3 = [B(3 * 1536) for _ in range(3)], [B(3 * D) for _ in range(3)]
        self.blm = [B(5120) for _ in range(24)]
        self.K = np.zeros((self.NL, max_ctx, NKV, HD), np.float32)
        self.V = np.zeros_like(self.K)
        inv = theta ** (-np.arange(0, HD, 2, dtype=np.float64) / HD)
        ang = np.arange(max_ctx)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        self.stats = {"npu_s": 0.0, "cpu_s": 0.0, "steps": 0}
        self.prefill = None
        if prefill_tokens:
            from .prefill import Prefill
            self.prefill = Prefill(self, max_tokens=prefill_tokens, verbose=verbose)

    def _run(self, chain):
        s = time.perf_counter()
        self.rt.run_chain(chain)
        self.stats["npu_s"] += time.perf_counter() - s

    def _ffn(self, l, x, ids, w):
        """Experts ``ids`` (indices into the layer's list) with weights ``w`` (None = all weight 1)."""
        np.copyto(self.bx.numpy(), _bf(x))
        if self.E == 3 and len(ids) == 9:
            gu, dn = self.Wgu[l], self.Wd[l]
            ch = [(self.h["g3"], [self.bgu3[j], self.bx] + [gu[e] for e in ids[3 * j:3 * j + 3]]) for j in range(3)]
            ch += [(self.h["d3"], [self.bd3[j], self.bgu3[j]] + [dn[e] for e in ids[3 * j:3 * j + 3]]) for j in range(3)]
            self._run(ch)
            y = np.concatenate([b.numpy().astype(np.float32).reshape(3, D) for b in self.bd3])
        else:
            ch = [(self.h["g"], [self.bgu[i], self.Wgu[l][e], self.bx]) for i, e in enumerate(ids)]
            ch += [(self.h["d"], [self.bd[i], self.Wd[l][e], self.bgu[i]]) for i, e in enumerate(ids)]
            self._run(ch)
            y = np.stack([b.numpy().astype(np.float32) for b in self.bd[:len(ids)]])
        if w is None:
            return y.sum(0)
        return (w[:, None] * y[:8]).sum(0) + y[8]

    def _rope(self, x, pos):
        c, s = self.cos[pos], self.sin[pos]
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def step(self, tok, pos, logits=True):
        """One token at position ``pos``; returns the next-token logits (None when ``logits`` is False:
        prompt tokens before the last only fill the KV cache, so their LM head is skipped)."""
        x = np.array(self.embed[tok], np.float32)
        for l in range(self.NL):
            L = self.L[l]
            np.copyto(self.bx.numpy(), _bf(_rms(x, L["input_layernorm"])))
            self._run([(self.h["q"], [self.bqkv, self.Wqkv[l], self.bx])])
            qkv = self.bqkv.numpy().astype(np.float32)
            s = time.perf_counter()
            q = self._rope(_rms(qkv[:NH * HD].reshape(NH, HD), L["q_norm"]), pos)
            k = self._rope(_rms(qkv[NH * HD:(NH + NKV) * HD].reshape(NKV, HD), L["k_norm"]), pos)
            self.K[l, pos], self.V[l, pos] = k, qkv[(NH + NKV) * HD:].reshape(NKV, HD)
            Kc, Vc = self.K[l, :pos + 1], self.V[l, :pos + 1]
            sc = np.einsum("hgd,thd->hgt", q.reshape(NKV, NH // NKV, HD), Kc) / np.sqrt(HD)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            attn = np.einsum("hgt,thd->hgd", sc, Vc).reshape(NH * HD)
            self.stats["cpu_s"] += time.perf_counter() - s
            np.copyto(self.ba[0].numpy(), _bf(attn[:2048]))
            np.copyto(self.ba[1].numpy(), _bf(attn[2048:]))
            self._run([(self.h["o"], [self.bo[j], self.Wo[l][j], self.ba[j]]) for j in range(2)])
            x = x + self.bo[0].numpy().astype(np.float32) + self.bo[1].numpy().astype(np.float32)
            xn = _rms(x, L["post_attention_layernorm"])
            if l == 0:
                x = x + self._ffn(0, xn, list(range(9)), None)
            else:
                s = time.perf_counter()
                p = 1.0 / (1.0 + np.exp(-(L["router"] @ xn)))
                ex = np.argsort(-(p + L["bias"]))[:TOPK]           # bias picks experts only
                w = p[ex] / max(p[ex].sum(), 6.103515625e-5) * self.scale
                self.stats["cpu_s"] += time.perf_counter() - s
                x = x + self._ffn(l, xn, list(ex) + [NE], w.astype(np.float32))
        self.stats["steps"] += 1
        if not logits:
            return None
        np.copyto(self.bx.numpy(), _bf(_rms(x, self.norm)))
        self._run([(self.h["q"], [self.blm[j], self.Wlm[j], self.bx]) for j in range(24)])
        return np.concatenate([b.numpy().astype(np.float32) for b in self.blm])[:self.vocab]

    def generate(self, prompt_ids, max_new, stop_ids=()):
        """Greedy. Returns (generated ids, seconds for the prompt, per-token decode seconds)."""
        t0 = time.perf_counter()
        if self.prefill is not None and len(prompt_ids) > 1:
            logits = self.prefill.run(prompt_ids)
        else:
            last = len(prompt_ids) - 1
            for pos, t in enumerate(prompt_ids):
                logits = self.step(t, pos, logits=pos == last)
        t_prompt = time.perf_counter() - t0
        out, per, pos = [], [], len(prompt_ids)
        for _ in range(max_new):
            t = int(np.argmax(logits))
            out.append(t)
            if t in stop_ids:
                break
            s = time.perf_counter()
            logits = self.step(t, pos)
            per.append(time.perf_counter() - s)
            pos += 1
        return out, t_prompt, per
