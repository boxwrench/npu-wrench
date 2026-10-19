# v0.3 — 2026-10-18

- `npuwrench/server.py`: background memory-writer service. Idle-gated scheduling (writes only after the iGPU has
  been idle `--idle-hold` s; pause at prefill-layer / decode-token boundaries; `--min-duty` bounds the backlog),
  async queue with per-session `received_through` / contiguous `extracted_through`, prompt-prefix KV reuse,
  OpenAI-compatible `/v1/chat/completions`. Engine: `generate(start=, hook=)` and prefill from a start position.
- Measured beside an interactive llama.cpp foreground: gated writes cost its decode 0.3 % (ungated 6.4 %).

# v0.2.2 — 2026-10-18

- KV cache stored position-major for K as well (contiguous slices for the attention matmuls): decode after a 2K
  prompt 105 -> 97 ms/token; short context 79.5 ms. Two 384-token writer outputs identical to v0.2.1; the check
  prompt now matches llama.cpp on 58 of 64 tokens (was 56: a near-tie flipped with the summation order).
- docs/METHOD.md: an NPU decode-attention operator was screened and not integrated (slower than the CPU at 2K).

# v0.2.1 — 2026-10-18

- Fix: decode attention used `np.einsum`, whose cost grew with context: ~305 ms/token after a 2K-token prompt.
  The KV cache is now laid out for batched matmuls: 105 ms/token after 2K tokens, 80.0 ms at short context
  (was 87.2). Check prompt still 56/64; one 384-token writer output identical, another identical for 124
  tokens then a near-tie greedy flip from the changed summation order.
- README: decode speed reported at short and long context; v0.2's 87 ms was short-context only.

# v0.2 — 2026-10-18

- NPU prefill (`--prefill N`): batched prompt processing with IRON's DequantBFP reading the decode weights in
  place (input pattern changed for npuwrench's layout; down padded K 768 -> 1024) and flm.GEMM; experts run on
  their routed tokens only. 2K-token prompt 18.0 s (was ~212 s token by token). Output unchanged (56/64).
- Skip HRX's per-dispatch cache flush for weight buffers (written once at load): decode 106.4 -> 87.2 ms/token.
- Prompt tokens before the last no longer run the LM head.
- CPU attention for prefill uses batched BLAS matmuls and masks only the diagonal block.

# v0.1.2 — 2026-10-18

- Fix: the activation's DMA task is freed with its round, so a dispatch is no longer capped near 10 rounds by the
  shim's 16 buffer descriptors. Output and speed unchanged (56/64 matching, 106.9 ms/token).
- docs/METHOD.md: per-token time breakdown, and the streaming experiments that did not help.

# v0.1.1 — 2026-10-18

- README: NPU map graphic, linked to the full measured map in REM and the xdna-top envelope.

# v0.1 — 2026-10-18 (early public release)

- Hy-Memory-30B-A3B decode end to end on the XDNA2 NPU: every linear layer as a fused q4nx GEMV in one
  resident program (two 4-column stages), attention and router on the CPU.
- Output check against llama.cpp Q4_K_M (`tools/check.py`, `results/`).
- Packer from the BF16 checkpoint (`tools/pack.py`); packed weights on Hugging Face.
- docs/METHOD.md: measured NPU costs (dispatch, program switch, streaming, buffer limits) and pitfalls.
