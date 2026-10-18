# Contributing

Help is welcome, especially from people with an XDNA2 laptop (Strix Point, Kraken Point) or more
mlir-aie experience than we have. Open an issue with what you measured and on which chip, or a PR.

## Where help would move the numbers most

Measured per-token cost today is listed in the README. In rough order of expected gain:

1. **Fewer dispatches per layer.** A dispatch carries about 5 buffers, so 9 experts need several
   dispatches. Keeping all of a layer's experts in one buffer and patching the expert offsets into
   the instruction stream per token (as MLIR-AIR's fused decode does with runtime state) would cut
   the MoE to 2 dispatches per layer.
2. **More DMA throughput.** The GEMV is data-movement bound at ~22 GB/s on 4 columns with one MM2S
   channel per column. Public W4A16 GEMV work on XDNA2 (TileFuse) implies ~45-52 GB/s. Using the
   second MM2S channel per shim, deeper weight FIFOs, or a pre-tiled interleaved layout are the
   candidates.
3. **NPU prefill.** Prompts currently go through the decode path one token at a time. IRON's
   `flm.GEMM` reaches 9-19 TF on this NPU at M=512-4096, and a grouped-expert prefill was prototyped
   at 0.37 s per MoE layer for an 8K prompt.
4. **Attention on the NPU.** Attention runs on the CPU. There is open head-dim-128 GQA prefill code
   for AIE2P in MLIR-AIR; we know of no open hd128 causal decode-attention kernel.
5. **Quality.** An 8-bit (or 6-bit) down projection would close most of the gap to llama.cpp
   Q4_K_M (docs/METHOD.md). A paired writer-quality suite is the planned acceptance test.
6. **Other models.** The stages only depend on the matrix shapes. MiniCPM-V-4.7-35B-A3B
   (Qwen3.6-MoE: 256 experts x 512, head dim 256) is the next target.

## Ground rules

- Greedy decoding for every comparison; report the chip, firmware, kernel, mlir-aie and IRON versions.
- A speed change must keep the output check passing (tools/check.py) and state its noise floor.
- Do not change NPU AXI QoS; hold the GPU fabric clock when the GPU and NPU run together.
