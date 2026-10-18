# How the NPU path works, and what we measured on the way

Everything here was measured on one machine: Ryzen AI Max+ 395 (Strix Halo), 128 GB, XDNA2 NPU
(firmware 1.1.2.65), mlir-aie 1.4.4, IRON `f280ccf`, HRX from AMD FastFlow-SDK. Numbers on Strix
Point or Kraken Point may differ.

## The kernel: AMD's open fused 4-bit GEMV

mlir-aie ships `q4nx_accumulate` (`aie_kernels/flm_gemma4/q4nx.h`): a matrix-vector product that
reads 4-bit q4nx weights (blocks of 32 outputs x 256 inputs, a bf16 scale and min per 32 inputs)
and dequantizes them in tile memory, so no 16-bit copy of the weights is ever written. IRON wraps it
as `flm/lm_head` (Gemma 4's LM head). We took that operator, removed its RMS norm and softcap, and
made the column count a parameter: the original requires `vocab % 1024 == 0` because one round is
8 columns x 4 rows x 32 outputs; with 4 columns a round is 512 outputs.

Decode on a padded GEMM path (one real row in a 256-row tile, weights dequantized to bfp16 first)
ran at 0.53 s/token for this model. The fused GEMV reads each weight once, at 5 bits.

## One resident program for the whole model

| Finding | Measured | What we did |
|---|---|---|
| Switching xclbins (programs) costs ~2-3 ms | two stages as two xclbins: 4.4 ms per token-layer of switching | put every stage in one design |
| Two handles on the *same* xclbin with different instruction streams alternate for free | -0.001 ms | one xclbin, one instruction stream per stage |
| Persistent workers on disjoint columns can share a program | gate/up on columns 0-3, down on 4-7 | stage picked by the instruction stream; idle stage waits on its FIFOs |
| Output size can live in the instruction stream | re-sending the activation each round costs nothing measurable (21.8 GB/s either way) | stage A serves every 2048-input matrix: qkv, o-proj halves, gate+up, LM-head blocks |
| SiLU(gate)·up can run on the down cores | one chain per layer, no host round trip | prologue in `q4nx_gemv.cc` using `tanh_bf16_v16`, as AMD's `swiglu.cc` |

All shim endpoints of every stage must be passed to the `Runtime`, or the unused stage fails with
`Prod endpoint not set`. Two input widths in one program need distinct kernel symbols (the `.cc` is
built with a `-DEXPERT_GEMV_SUFFIX`).

## Where the time goes

| Item | Measured |
|---|---|
| Fixed cost per chained HRX dispatch | ~18 us (single dispatch 66 us) |
| Weight streaming, 4 columns, one MM2S channel each | ~22 GB/s on 9-expert chains, 28 GB/s on one large matrix |
| Same, 2 columns | 16 GB/s |
| Same program with the multiply skipped on the cores | identical time: the GEMV is data-movement bound |
| Buffer arguments per dispatch | about 5 (with 9 weight buffers only the first 3 arrive; no error is raised) |

So the levers are fewer dispatches and more DMA throughput, not core compute.

## Where a decode token goes (v0.1, 3 experts per dispatch)

| Part | ms | Weights read | Effective rate |
|---|---:|---:|---:|
| MoE (47 layers + the dense first layer) | 54.4 | ~1.27 GB | 23.5 GB/s |
| o-proj (two halves per layer) | 15.5 | 252 MB | 16 GB/s |
| qkv | 15.0 | 315 MB | 21 GB/s |
| LM head | 5.4 | 155 MB | 29 GB/s |
| CPU attention + router | 8.0 | | |
| Other host work (Python) | 4.9 | | |
| **Total** | **103** | ~2.0 GB | |

## What did not raise the streaming rate

All on random weights, same program, chains of 9 experts; the control was repeated to get the floor
(about 1 %).

| Change | gate/up x9 | down x9 | Verdict |
|---|---:|---:|---|
| Control (FIFO depth 2, 2 DMA tasks in flight) | 0.815-0.829 ms | 0.550-0.563 ms | |
| 4 DMA tasks in flight | 0.815 | 0.549 | within noise |
| FIFO depth 3 | — | — | does not compile: the memory tile runs out of BD ids (24 per channel) with a 4-way row split |
| 2-4 weight blocks per core handshake | 0.855-0.915 | 0.580 | slower |
| 6 or 8 columns instead of 4 (earlier) | ~5 % faster | ~3 % faster | small |

The core is not waiting on locks or on DMA tasks in flight, and adding columns barely helps, so the
limit looks like a shared NPU-to-memory ceiling of about 22-28 GB/s for this access pattern. Getting
toward the ~45-52 GB/s that public W4 GEMV work implies (TileFuse, Point/Kraken) likely needs a
different weight path, not tuning of this one.

- **Shim BD budget.** A shim tile has 16 buffer descriptors. Re-sending the activation every round
  must free each round's activation transfer with that round's task group; holding them to the end
  of the dispatch caps a dispatch at about 10 rounds (fixed in v0.1.2).

## Prefill

| Prompt | Total | NPU | CPU |
|---:|---:|---:|---:|
| 512 tokens | 7.1 s | 4.6 s | 2.5 s |
| 2048 tokens | 18.0 s | 5.3 s | 12.7 s |

- `flm.GEMM` takes M, K and N at run time and `DequantBFP` builds its xclbin from a fixed reference shape,
  so prefill uses two programs whatever the shapes. M must be a multiple of 256; expert row counts are
  rounded up to powers of two to keep the number of instruction streams small.
- Our q4nx blocks are byte-identical to DequantBFP's; only the block order differs, by an affine map. The
  block of out-block `nb` and k-tile `k` sits at `(nb // 4 * kb + k) * 4 + nb % 4`, so DequantBFP's column
  block `cb` starts at block `(cb // 2) * 4 * kb + 2 * (cb % 2)` with a k stride of 4 blocks. Changing the
  fill pattern reads the decode weights in place.
- The down projection (K = 768) is dequantized as K = 1024: the extra k-tile reads finite neighbouring
  blocks (a 4-block zero tail covers the last one) and multiplies zero-padded activations.
- NPU time barely grows with the prompt (4.6 s at 512 tokens, 5.3 s at 2048); the CPU's causal attention
  is now most of the time.

## Cache maintenance is not free

HRX flushes every argument buffer's cache lines before each dispatch and invalidates outputs before a host
read. For weights that never change after loading this is pure overhead: skipping it took decode from
106.4 to 87.2 ms/token and halved prefill's NPU time. Buffers written and read only by the NPU (the bfp16
GEMM operands) need neither; host buffers need it only over the rows actually used.

## Quality of the 4-bit format

Scored on one MoE layer (layer 1, 32 token rows) against a float64 reference computed from the BF16
weights, same inputs and router:

| Weights | Routed output error (median rel. L2) |
|---|---:|
| llama.cpp Q4_K_M (down in Q6_K) | 6.7 % |
| q4nx, plain min/max encoder | 10.4 % |
| q4nx, least-squares scale/min search (shipped) | 8.8 % |
| + activation-weighted search | 8.6 % |
| + BF16 down (upper bound for a higher-precision down) | 6.2 % |

The kernel itself adds almost nothing (live NPU output matches a CPU emulation of the same weights).
Most of the remaining gap to Q4_K_M is the 4-bit down projection.

## Pitfalls

- `amdxdna ... Alloc hw resource failed` (errno 22) appeared when one chain touched many distinct
  executables; HRX caches 16 executables per process by default. The one-program design avoids it.
- Strix Halo: hold the GPU fabric clock at its top level while the GPU and NPU are both active
  (reported by Halogen; we run with it held). Do not raise NPU AXI QoS.
- `pytest` collection in IRON skips AIE2P tests without a device bound; call the test functions
  directly. HRX needs `NPU_RUNTIME=hrx`.
