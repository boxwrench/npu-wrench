# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""Minimal safetensors read/write (U8, F32, BF16) without the safetensors package."""
import json
import mmap
import struct

import numpy as np

_DT = {"U8": np.uint8, "F32": np.float32, "BF16": np.uint16, "I32": np.int32}
_NAME = {np.dtype(np.uint8): "U8", np.dtype(np.float32): "F32", np.dtype(np.int32): "I32"}


def save(path, tensors, bf16=()):
    """tensors: name -> ndarray. Names in ``bf16`` are float32 arrays stored as BF16 (round to nearest even)."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        if name in bf16:
            u = a.astype(np.float32).view(np.uint32)
            a = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            dt = "BF16"
        else:
            dt = _NAME[a.dtype]
        b = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


class Reader:
    """Memory-mapped reader. ``get`` returns a read-only view; BF16 comes back as float32."""

    def __init__(self, path):
        self._f = open(path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n))
        self.header.pop("__metadata__", None)
        self._base = 8 + n
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def names(self):
        return list(self.header)

    def get(self, name):
        m = self.header[name]
        a, b = m["data_offsets"]
        arr = np.frombuffer(self._mm, dtype=_DT[m["dtype"]], count=(b - a) // np.dtype(_DT[m["dtype"]]).itemsize,
                            offset=self._base + a).reshape(m["shape"])
        if m["dtype"] == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr
