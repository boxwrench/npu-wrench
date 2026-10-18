---
license: apache-2.0
base_model: tencent/Hy-Memory-30B-A3B
base_model_relation: quantized
library_name: npuwrench
tags:
- amd
- npu
- xdna2
- strix-halo
- ryzen-ai
- moe
- memory
---

# Hy-Memory-30B-A3B — q4nx for the AMD XDNA2 NPU

Packed weights of [tencent/Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B) for
running every linear layer on the AMD XDNA2 NPU (Ryzen AI Max / Strix Halo; Strix Point and Kraken
Point untested). The runtime, packer and measurements are at [boxwrench/npu-wrench](https://github.com/boxwrench/npu-wrench).

> **Status, 2026-10-18: early release.** Decode runs end to end and matches llama.cpp's greedy output
> on the check prompt. v0.2 adds NPU prompt processing (prefill); attention runs on the CPU.

## Base model

Hy-Memory-30B-A3B is Tencent's memory-writer model: 48 layers, hidden size 2048, 128 routed experts
(top 8, width 768) plus one shared expert, GQA attention with 32 query and 4 KV heads of dimension
128. It is trained to write long-term memories inside existing agent-memory frameworks (Mem0, MemOS
and others). Apache-2.0; see the [model card](https://huggingface.co/tencent/Hy-Memory-30B-A3B).

## This format

- **q4nx**, the 4-bit format of AMD's open `q4nx_accumulate` NPU kernel: blocks of 32 outputs by 256
  inputs, 4-bit codes with a bf16 scale and minimum per 32 inputs (5 bits per weight), stored in the
  block order the runtime's 4-column stages stream.
- Quantized from the BF16 checkpoint with a least-squares scale/minimum search per group. No
  calibration data was used.
- Every linear layer is q4nx, including the expert down projections and the LM head. The embedding
  is BF16; norms, router and expert bias are float32.
- Files: `layer_00..47.safetensors`, `head.safetensors` (LM head blocks, embedding, final norm),
  `config.json`, and the model's own tokenizer and chat template. About 19.5 GB.

These files are not GGUF and are not loadable by llama.cpp, vLLM or transformers.

## Speed, one Strix Halo

<!-- BENCH:START -->
| Path | Decode tok/s | ms/token |
|---|---:|---:|
| NPU, v0.2 (3 experts per dispatch) | 11.5 | 87.2 |

| Prompt (NPU prefill, v0.2) | Time |
|---:|---:|
| 512 tokens | 7.1 s |
| 2048 tokens | 18.0 s |
<!-- BENCH:END -->

Greedy, 64 generated tokens after a 63-token prompt. Receipt and method in the GitHub repository.

## Quality

The first 56 of 64 greedy tokens on the check prompt match llama.cpp's Q4_K_M output exactly. Per layer,
against a BF16 reference, the routed-expert output error is 8.8 % (llama.cpp Q4_K_M: 6.7 %), mostly
from the 4-bit down projection. A task-level writer evaluation has not been run yet.

## Run it

```sh
git clone https://github.com/boxwrench/npu-wrench
hf download boxwrench/Hy-Memory-30B-A3B-q4nx-wrench --local-dir ./Hy-Memory-30B-A3B-q4nx-wrench
python tools/run.py --weights ./Hy-Memory-30B-A3B-q4nx-wrench --chat examples/extract.json --max-new 256 \
    --experts-per-dispatch 3 --prefill 4096
```

Setup (mlir-aie, IRON, HRX) is in the repository README.

## Credits and license

- Base model: [tencent/Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B), Apache-2.0,
  Copyright Tencent. Its licence is included as `LICENSE-Hy-Memory`; the tokenizer and chat template
  are redistributed unchanged.
- NPU kernel and operators: mlir-aie and amd/IRON (AMD, Apache-2.0); runtime: HRX (AMD FastFlow-SDK).
- These weights are a quantized derivative of Hy-Memory-30B-A3B under Apache-2.0.
