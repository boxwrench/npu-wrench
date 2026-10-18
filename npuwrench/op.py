# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
# Derived from amd/IRON's MLIROperator pattern (iron/operators/flm/lm_head/op.py).
"""One instruction stream of the two-stage universal q4nx GEMV program."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import aie.utils as aie_utils
from iron.common import AIERuntimeArgSpec, DesignGenerator, KernelObjectArtifact, MLIROperator, PythonGeneratedMLIRArtifact
from iron.operators.flm.q4nx import packed_bytes
from .kernel import kernel


@dataclass
class UniGEMV(MLIROperator):
    which: str
    n_out: int
    E: int = 1
    wdepth: int = 2
    win: int = 2
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        self.dim = 2048 if self.which == "A" else 768
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self):
        return f"Uni_{self.which}{self.n_out}" + (f"_E{self.E}" if self.E > 1 else "") + (f"_w{self.wdepth}_{self.win}" if (self.wdepth, self.win) != (2, 2) else "") + f"_{aie_utils.get_current_device().resolve().name}"

    def quantized_size(self):
        return packed_bytes(self.n_out * self.dim)

    def _kerns(self):
        return {(2048, False): kernel(2048), (768, True): kernel(768, True)}

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(Path(__file__).resolve().parent / "design.py", "uni_gemv" if self.E == 1 else "uni_multi",
                            (aie_utils.get_current_device(), self.which, self.n_out) + ((self.E,) if self.E > 1 else ()) + (self.wdepth, self.win),
                            {"kerns": self._kerns()}))

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(k) for k in self._kerns().values()]

    def get_arg_spec(self):
        xin = 1536 if self.which == "D" else 2048
        if self.E > 1:
            nx = self.E if self.which == "D" else 1
            return ([AIERuntimeArgSpec("out", (self.E * self.n_out,), dtype=bfloat16),
                     AIERuntimeArgSpec("in", (nx * xin,), dtype=bfloat16)]
                    + [AIERuntimeArgSpec("in", (self.quantized_size() // 4,), dtype=np.uint32)] * self.E)
        return [AIERuntimeArgSpec("out", (self.n_out,), dtype=bfloat16),
                AIERuntimeArgSpec("in", (self.quantized_size() // 4,), dtype=np.uint32),
                AIERuntimeArgSpec("in", (xin,), dtype=bfloat16)]
