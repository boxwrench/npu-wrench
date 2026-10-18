# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Prefill operators: IRON's DequantBFP reading npuwrench's GEMV weight layout, and the flm.GEMM it feeds."""
from dataclasses import dataclass
from pathlib import Path

import aie.utils as aie_utils
from iron.common import DesignGenerator, PythonGeneratedMLIRArtifact
import iron.operators.flm.dequant.op as _dq
from iron.operators.flm.dequant.op import DequantBFP
from iron.operators.flm.q4nx import BLOCK_BYTES

ROWS = 4
DESIGN = Path(__file__).resolve().parent / "dequant_design.py"


@dataclass
class DequantLayout(DequantBFP):
    """q4nx in npuwrench's layout (``src_kb`` k-tiles per out-block) to the bfp16 B that flm.GEMM reads.

    ``K`` may exceed ``256 * src_kb`` (padding K to a multiple of 512); the input buffer must then hold
    ``src_bytes`` (``tail_blocks`` extra blocks), see ``dequant_design.py``.
    """
    src_kb: int = None

    @property
    def operator_dir(self):
        # The shape-independent config xclbin is built from IRON's own design.py, unchanged.
        return Path(_dq.__file__).resolve().parent

    @staticmethod
    def tail_blocks(K, N, src_kb):
        """Blocks the fill reads past the end of an N x (256 * src_kb) blob."""
        n_cb = N // 64
        last = ((n_cb - 1) // 2) * ROWS * src_kb + 2 * ((n_cb - 1) % 2) + (K // 256 - 1) * ROWS + 1
        return max(0, last + 1 - (N // 32) * src_kb)

    @property
    def src_bytes(self):
        return ((self.N // 32) * self.src_kb + self.tail_blocks(self.K, self.N, self.src_kb)) * BLOCK_BYTES

    @property
    def name(self):
        return f"NW_DequantLayout_K{self.K}_N{self.N}_src{self.src_kb}_{self._config_tag}"

    def quantized_size(self):
        return self.src_bytes

    def get_mlir_artifact(self):
        # The per-shape instruction stream reads our layout; the config xclbin stays DequantBFP's.
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                DESIGN, "dequant_bfp",
                (aie_utils.get_current_device(), self.K, self.N, self.tile_n, None, None, 0,
                 self.src_kb, self.src_bytes),
                {"dequant_kernel": self._kernel()},
            ),
        )
