#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Pack tencent/Hy-Memory-30B-A3B (HF BF16) into NPU weight files for npuwrench.

Every linear layer becomes q4nx (4-bit codes, bf16 scale + min per 32 inputs, 5 bits/weight) in the
block order the 4-column stages read, with a least-squares scale/min search per group (encoder.py).
Norms, router and expert bias stay float32; the embedding stays BF16.

usage: python tools/pack.py --hf /path/to/Hy-Memory-30B-A3B --out /path/to/Hy-Memory-30B-A3B-q4nx-wrench [--procs 16]
Output: config.json, layer_00..47.safetensors, head.safetensors (~19.5 GB). One layer per process.
"""
import argparse
import json
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from npuwrench import st                      # noqa: E402
from npuwrench.encoder import encode_layout_fast  # noqa: E402

COLS, ROWS = 4, 4
HF = None


class _BF16:
    def __init__(self, root):
        self.root = root
        self.idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        f = self.idx[name]
        if f not in self.h:
            fh = open(os.path.join(self.root, f), "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            self.h[f] = (fh, json.loads(fh.read(n)), 8 + n)
        fh, h, base = self.h[f]
        m = h[name]
        a, b = m["data_offsets"]
        fh.seek(base + a)
        raw = fh.read(b - a)
        if m["dtype"] == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
        return np.frombuffer(raw, np.float32).reshape(m["shape"]).copy()


def enc(w_out_in):
    n, dim = w_out_in.shape
    return encode_layout_fast(np.ascontiguousarray(w_out_in.T), dim, n, COLS, ROWS)


def pack_layer(args):
    l, out = args
    path = os.path.join(out, f"layer_{l:02d}.safetensors")
    if os.path.exists(path):
        return l
    g = _BF16(HF).get
    p = f"model.layers.{l}."
    t = {}
    q, k, v = (g(p + f"self_attn.{x}_proj.weight") for x in "qkv")
    t["qkv"] = enc(np.concatenate([q, k, v]))
    o = g(p + "self_attn.o_proj.weight")
    t["o0"], t["o1"] = enc(o[:, :2048]), enc(o[:, 2048:])
    for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
        t[name.split(".")[-1]] = g(p + name + ".weight").astype(np.float32)
    if l == 0:
        gw, uw, dw = (g(p + f"mlp.{x}_proj.weight") for x in ("gate", "up", "down"))
        for j in range(9):
            s = slice(j * 768, (j + 1) * 768)
            t[f"C{j}_gu"] = enc(np.concatenate([gw[s], uw[s]]))
            t[f"C{j}_d"] = enc(dw[:, s])
    else:
        t["router"] = g(p + "mlp.router.gate.weight").astype(np.float32)
        t["bias"] = g(p + "mlp.expert_bias").astype(np.float32)
        for e in list(range(128)) + ["s"]:
            pe = p + ("mlp.shared_mlp." if e == "s" else f"mlp.experts.{e}.")
            gw, uw, dw = (g(pe + f"{x}_proj.weight") for x in ("gate", "up", "down"))
            t[f"E{e}_gu"] = enc(np.concatenate([gw, uw]))
            t[f"E{e}_d"] = enc(dw)
    st.save(path + ".tmp", t)
    os.replace(path + ".tmp", path)
    return l


def main():
    global HF
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    HF = a.hf
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.hf, "config.json")))
    assert cfg["model_type"] == "hy_v3" and cfg["hidden_size"] == 2048 and cfg["num_experts"] == 128
    t0 = time.time()
    head = os.path.join(a.out, "head.safetensors")
    if not os.path.exists(head):
        g = _BF16(a.hf).get
        lm = g("lm_head.weight")
        lm = np.concatenate([lm, np.zeros((24 * 5120 - lm.shape[0], lm.shape[1]), np.float32)])
        t = {f"lm_{j}": enc(lm[j * 5120:(j + 1) * 5120]) for j in range(24)}
        t["embed"] = g("model.embed_tokens.weight")
        t["norm"] = g("model.norm.weight").astype(np.float32)
        st.save(head + ".tmp", t, bf16=("embed",))
        os.replace(head + ".tmp", head)
    with Pool(a.procs) as pool:
        for l in pool.imap_unordered(pack_layer, [(l, a.out) for l in range(cfg["num_hidden_layers"])]):
            print(f"layer {l} done ({time.time() - t0:.0f}s)", flush=True)
    meta = {k: cfg[k] for k in ("num_hidden_layers", "vocab_size", "router_scaling_factor", "hidden_size",
                                 "num_experts", "num_experts_per_tok", "num_attention_heads", "num_key_value_heads",
                                 "head_dim", "rms_norm_eps")}
    meta["rope_theta"] = cfg["rope_parameters"]["rope_theta"]
    meta["format"] = {"name": "Hy-Memory-30B-A3B-q4nx-wrench", "version": 1, "cols": COLS, "rows": ROWS,
                      "encoder": "q4nx, least-squares scale/min search, no activation weighting"}
    json.dump(meta, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(a.hf, f)
        if os.path.exists(src):
            with open(src, "rb") as s, open(os.path.join(a.out, f), "wb") as d:
                d.write(s.read())
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
