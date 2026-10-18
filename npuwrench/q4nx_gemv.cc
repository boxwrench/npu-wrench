//===- q4nx_gemv.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Modifications Copyright (C) 2026 boxwrench: per-dim symbol suffix, SwiGLU prologue.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One MoE expert projection as a q4nx matrix-vector product: accumulate the
// 4-bit weights straight from L3/DDR in tile memory. The RMS norm and the
// softcap of q4nx_lm_head.cc are gone; the caller supplies the activation.
// Geometry arrives as -D flags (Q4NX_M_TILE, Q4NX_K_TILE, Q4NX_GROUP,
// EXPERT_GEMV_DIM = the number of in-features).
#include "aie_kernels/aie_kernel_utils.h"
#include "aie_kernels/common/zero.h"
#include "aie_kernels/flm_gemma4/q4nx.h"
#include "aie_kernels/flm_gemma4/utils.h"

#include <aie_api/aie.hpp>
#ifdef EXPERT_GEMV_SWIGLU
#include "aie_kernels/common/activations.h"
#endif
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) || \
    !defined(EXPERT_GEMV_DIM)
#error "expert_gemv.cc needs its geometry -D defined"
#endif

using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int K_BLOCKS = EXPERT_GEMV_DIM / K_TILE;

static_assert(Q4NX_GROUP == 32, "a column sum covers exactly one group");
static_assert(EXPERT_GEMV_DIM % K_TILE == 0,
              "the block loop covers whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");

} // namespace

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define SYM(n) CAT(n, EXPERT_GEMV_SUFFIX)

extern "C" {

/// Once per dispatch: the per-32 column sums of x. q4nx_accumulate folds each
/// block's minimum into one MAC per group with them.
void SYM(expert_gemv_sums)(const bfloat16 *x, bfloat16 *b_group_sums) {
  event0();
  for (int k = 0; k < K_BLOCKS; k++) {
    group_sums_32<K_TILE>(b_group_sums + k * (K_TILE / 32), x + k * K_TILE);
  }
  event1();
}

/// Before a tile's k loop.
void SYM(expert_gemv_zero)(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

/// One 32 x 256 weight block: y_acc += dequantize(w) * x[k * 256 ...].
void SYM(expert_gemv_block)(const q4nx_block_t *w, const bfloat16 *x, float *y_acc,
                       const bfloat16 *b_group_sums, int k) {
  event0();
  q4nx_accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc,
                                  b_group_sums + k * (K_TILE / 32));
  event1();
}

/// After the k loop: the float accumulator narrowed to bf16, no epilogue math.
void SYM(expert_gemv_epilogue)(bfloat16 *y, const float *y_acc) {
  event0();
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

#ifdef EXPERT_GEMV_SWIGLU
// MoE down prologue. gu = [gate(DIM) | up(DIM)] from stage A; h = silu(gate) * up,
// silu(g) = g * (0.5 + 0.5 * tanh(g / 2)) as aie_kernels/activation/swiglu.cc; then the
// per-32 sums of h, as expert_gemv_sums.
void SYM(expert_gemv_swiglu_sums)(const bfloat16 *gu, bfloat16 *h,
                                  bfloat16 *b_group_sums) {
  event0();
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> half_acc;
  half_acc.from_vector(half);
  for (int i = 0; i < EXPERT_GEMV_DIM; i += 16) {
    aie::vector<bfloat16, 16> g = aie::load_v<16>(gu + i);
    aie::vector<bfloat16, 16> u = aie::load_v<16>(gu + EXPERT_GEMV_DIM + i);
    aie::vector<bfloat16, 16> t =
        tanh_bf16_v16(aie::mul(g, half).template to_vector<bfloat16>());
    aie::vector<bfloat16, 16> sig =
        aie::mac(half_acc, t, half).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> silu =
        aie::mul(aie::max(g, bfloat16(-8.0f)), sig).template to_vector<bfloat16>();
    aie::store_v(h + i, aie::mul(silu, u).template to_vector<bfloat16>());
  }
  for (int k = 0; k < K_BLOCKS; k++) {
    group_sums_32<K_TILE>(b_group_sums + k * (K_TILE / 32), h + k * K_TILE);
  }
  event1();
}
#endif
}
