# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
# Derived from mlir-aie's flm_gemma4 kernel factories; symbols suffixed per dim.
"""q4nx GEMV kernel factory: symbols suffixed by dim (and 's' for the SwiGLU prologue)."""
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from iron.operators.flm.q4nx import BLOCK_BYTES, GROUP, K_TILE, M_TILE

_BF16, _F32 = np.dtype[bfloat16], np.dtype[np.float32]
SOURCE = Path(__file__).resolve().parent / "q4nx_gemv.cc"
_CACHE = {}


def kernel(dim, swiglu=False):
    key = (dim, swiglu)
    if key in _CACHE:
        return _CACHE[key]
    sfx = f"_d{dim}" + ("s" if swiglu else "")
    x = np.ndarray[(dim,), _BF16]; sums = np.ndarray[(dim // 32,), _BF16]
    y_acc = np.ndarray[(M_TILE,), _F32]; y_blk = np.ndarray[(M_TILE,), _BF16]
    w_blk = np.ndarray[(BLOCK_BYTES // 2,), _BF16]
    fn = ExternalFunction(
        "expert_gemv_block" + sfx, source_file=str(SOURCE),
        arg_types=[w_blk, x, y_acc, sums, np.int32], include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DQ4NX_M_TILE={M_TILE}", f"-DQ4NX_K_TILE={K_TILE}", f"-DQ4NX_GROUP={GROUP}",
                       f"-DEXPERT_GEMV_DIM={dim}", f"-DEXPERT_GEMV_SUFFIX={sfx}",
                       "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"] + (["-DEXPERT_GEMV_SWIGLU"] if swiglu else []))
    bind = fn.object_file.bind
    fn.k_block = fn
    fn.k_sums = bind("expert_gemv_sums" + sfx, [x, sums])
    fn.k_zero = bind("expert_gemv_zero" + sfx, [y_acc])
    fn.k_epi = bind("expert_gemv_epilogue" + sfx, [y_blk, y_acc])
    if swiglu:
        fn.k_swiglu = bind("expert_gemv_swiglu_sums" + sfx, [np.ndarray[(2 * dim,), _BF16], x, sums])
    _CACHE[key] = fn
    return fn
