# Contributing

Help is welcome, especially from people with an XDNA2 laptop (Strix Point, Kraken Point) or more
mlir-aie experience than we have. Open an issue with what you measured and on which chip, or a PR.

## Where help would matter most

The measurements so far (README, docs/METHOD.md) say the NPU writer's value is predictable interference that
a scheduler can hide, not raw speed: its cost to an iGPU foreground is shared memory bandwidth, so cutting the
bytes per write or writing in idle gaps helps more than a faster kernel. In that order:

**Use and quality**

1. **Other XDNA2 machines.** Strix Point and Kraken Point are untested: run `tools/check.py` and report the
   chip, firmware, kernel, mlir-aie and IRON versions.
2. **Writer quality.** A paired memory-writer suite (spans with prior memories in, expected memory changes
   out) is the planned acceptance test. A higher-precision down projection (6- or 8-bit, needs a q8 GEMV
   variant) would close most of the per-layer gap to llama.cpp Q4_K_M.
3. **Fewer bytes per write.** The writer over-stores (replies run past 300 tokens). A protocol that passes
   existing memories in and asks only for changes would cut decode, which is most of a write.
4. **Scheduling.** Untested: a foreground that never idles (`--min-duty`). Splitting prefill layers into
   smaller pieces would shrink the remaining ~0.38 s overlap when a foreground request arrives.

**Speed**

5. **More DMA throughput.** The GEMV is data-movement bound at ~22 GB/s on 4 columns, one MM2S channel per
   column; depth, window, blocks per handshake and column count were all flat. Public W4A16 GEMV work on
   XDNA2 (TileFuse) implies ~45-52 GB/s: a different weight path (pre-tiled interleaved layout, both MM2S
   channels per shim), not tuning of this one.
6. **Attention on the NPU.** Attention runs on the CPU (12.7 of 18 s of a 2K prefill). A first
   flash-decoding operator is correct to 256 positions and 1.5x slower than the CPU at 2K (docs/METHOD.md);
   MLIR-AIR has open hd128 GQA prefill code.
7. **A multi-row q4nx kernel on the AIE mmul unit.** Repeating the vector GEMV per row is compute-bound past
   2 rows; a 4-row matrix-multiply tile would make speculative verification and batching writes pay off.
8. **Fewer dispatches per layer.** A dispatch carries about 5 buffers; one buffer per layer with the expert
   offsets patched into the instruction stream per token would cut the MoE to 2 dispatches per layer.

**Other models**

9. The stages only depend on matrix shapes. MiniCPM-V-4.7-35B-A3B (Qwen3.6-MoE: 256 experts x 512, head
   dim 256) is the next target.

## Ground rules

- Greedy decoding for every comparison; report the chip, firmware, kernel, mlir-aie and IRON versions.
- A speed change must keep the output check passing (tools/check.py) and state its noise floor.
- Report foreground cost as tail latency (p95/p99, first token), not only averages.
- Do not change NPU AXI QoS; hold the GPU fabric clock when the GPU and NPU run together.
