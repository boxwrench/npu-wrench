#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Output check: greedy tokens from the NPU engine vs a reference run (llama.cpp Q4_K_M by default).

  python tools/check.py --weights DIR [--ref results/reference-llamacpp.json] [--n 64]

The reference file holds the prompt token ids and the reference greedy ids. Reports the number of
leading tokens that match, top-1 agreement over the run, and decode speed. Exit code 1 if fewer
than --min-prefix leading tokens match.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from npuwrench.engine import HyMemoryNPU  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", default=os.path.join(here, "..", "results", "reference-llamacpp.json"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--min-prefix", type=int, default=32)
    ap.add_argument("--experts-per-dispatch", type=int, default=1, choices=(1, 3))
    ap.add_argument("--prefill", type=int, default=0, help="max prompt tokens for NPU prefill (0 = token by token)")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    ids, want = ref["prompt_ids"], ref["gen_ids"][:a.n]
    eng = HyMemoryNPU(a.weights, max_ctx=max(len(ids) + a.n + 8, a.prefill), experts_per_dispatch=a.experts_per_dispatch,
                      prefill_tokens=a.prefill)
    got, t_prompt, per = eng.generate(ids, len(want))
    prefix = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), len(want))
    res = {"prompt_tokens": len(ids), "generated": len(got), "matching_prefix": prefix,
           "top1_agreement": round(float(np.mean([x == y for x, y in zip(got, want)])), 4),
           "decode_ms_per_token_median": round(float(np.median(per)) * 1e3, 1),
           "decode_tok_s": round(1.0 / float(np.median(per)), 2), "prompt_s": round(t_prompt, 2),
           "npu_s": round(eng.stats["npu_s"], 2), "cpu_s": round(eng.stats["cpu_s"], 2),
           "experts_per_dispatch": a.experts_per_dispatch, "reference": ref.get("reference", "")}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump({**res, "gen_ids": got}, open(a.out, "w"), indent=1)
    sys.exit(0 if prefix >= a.min_prefix else 1)


if __name__ == "__main__":
    main()
