# v0.1.1 — 2026-10-18

- README: NPU map graphic, linked to the full measured map in REM and the xdna-top envelope.

# v0.1 — 2026-10-18 (early public release)

- Hy-Memory-30B-A3B decode end to end on the XDNA2 NPU: every linear layer as a fused q4nx GEMV in one
  resident program (two 4-column stages), attention and router on the CPU.
- Output check against llama.cpp Q4_K_M (`tools/check.py`, `results/`).
- Packer from the BF16 checkpoint (`tools/pack.py`); packed weights on Hugging Face.
- docs/METHOD.md: measured NPU costs (dispatch, program switch, streaming, buffer limits) and pitfalls.
