# Export manifest

A fresh repository with no history from the development tree, exported 2026-10-18.

## Included

| Path | Why |
|---|---|
| `npuwrench/` | NPU kernel (`q4nx_gemv.cc`), IRON design and operator, q4nx layout and encoder, weight file reader, engine, prefill (`prefill.py`, `prefill_ops.py`, `dequant_design.py`), writer server (`server.py`) |
| `tools/` | `pack.py` (BF16 checkpoint to NPU weights), `run.py` (generate), `check.py` (output check vs llama.cpp) |
| `examples/` | the check prompt and a chat example |
| `results/` | llama.cpp reference tokens for the check, and our measured check results |
| `docs/METHOD.md`, `docs/assets/` | how the NPU path works and what was measured; the NPU map graphic |
| `hf/README.md` | the Hugging Face model card |
| `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `CHANGELOG.md` | licence, upstream notices, contribution guide |

## Excluded

Development scripts and probes, internal notes and experiment logs, machine-specific paths and
configuration, model weights and activation dumps, and prototypes that the shipped design replaced
(the padded-GEMM prefill prototype, the two-program layout).
