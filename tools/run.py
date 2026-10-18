#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Generate with Hy-Memory on the NPU (greedy).

  python tools/run.py --weights DIR --chat examples/extract.json --max-new 256
  python tools/run.py --weights DIR --text "raw continuation prompt" --max-new 64

--chat takes a JSON list of {"role", "content"} messages rendered with the model's chat template
(reasoning_effort no_think, the template default). Prints the reply and the timing.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from npuwrench.engine import HyMemoryNPU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--chat")
    ap.add_argument("--text")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--experts-per-dispatch", type=int, default=1, choices=(1, 3))
    ap.add_argument("--json-out")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.weights, "tokenizer.json"))
    if a.chat:
        import jinja2
        tpl = jinja2.Environment().from_string(open(os.path.join(a.weights, "chat_template.jinja")).read())
        prompt = tpl.render(messages=json.load(open(a.chat)), add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False).ids
    else:
        ids = tok.encode(a.text or "").ids
    eos = [i for i in (tok.token_to_id("<eos:6124c78e>"),) if i is not None]
    eng = HyMemoryNPU(a.weights, max_ctx=len(ids) + a.max_new + 8, experts_per_dispatch=a.experts_per_dispatch)
    out, t_prompt, per = eng.generate(ids, a.max_new, stop_ids=eos)
    text = tok.decode([t for t in out if t not in eos])
    print(text)
    dec = float(np.median(per)) if per else float("nan")
    res = {"prompt_tokens": len(ids), "generated_tokens": len(out), "prompt_s": round(t_prompt, 2),
           "decode_ms_per_token_median": round(dec * 1e3, 1), "decode_tok_s": round(1 / dec, 2) if per else None,
           "npu_s": round(eng.stats["npu_s"], 2), "cpu_attention_router_s": round(eng.stats["cpu_s"], 2)}
    print(json.dumps(res), file=sys.stderr)
    if a.json_out:
        json.dump({"ids": out, "text": text, **res}, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
