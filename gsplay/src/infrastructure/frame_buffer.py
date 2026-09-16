"""Bounded Windows shared-memory frame transport; the pipe carries descriptors only."""

import mmap
import uuid

import numpy as np


class FrameBuffer:
    def __init__(self, size: int, name: str | None = None):
        if not isinstance(size, int) or not 0 < size <= 2 * 1024**3:
            raise ValueError("Invalid shared frame capacity")
        self.size = size
        self.name = name or "gsplay-frame-" + uuid.uuid4().hex
        self.mapping = mmap.mmap(-1, size, tagname=self.name)

    def write(self, arrays: dict) -> dict:
        offset = 0
        fields = {}
        for name, array in arrays.items():
            array = np.ascontiguousarray(array)
            if array.dtype not in (np.dtype("float32"), np.dtype("bool")):
                raise ValueError("Unexpected decoded frame dtype")
            if offset + array.nbytes > self.size:
                raise ValueError("Decoded frame exceeds shared buffer capacity")
            target = np.ndarray(array.shape, dtype=array.dtype, buffer=self.mapping, offset=offset)
            target[...] = array
            fields[name] = [array.dtype.str, list(array.shape), offset]
            offset += array.nbytes
        return fields

    def read(self, fields: dict) -> dict:
        arrays = {}
        for name, (dtype, shape, offset) in fields.items():
            dtype = np.dtype(dtype)
            if dtype not in (np.dtype("float32"), np.dtype("bool")):
                raise ValueError("Invalid shared frame dtype")
            if not shape or any(not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError("Invalid shared frame shape")
            count = 1
            for n in shape:
                count *= n
            if (
                not isinstance(offset, int)
                or offset < 0
                or offset + count * dtype.itemsize > self.size
            ):
                raise ValueError("Invalid shared frame bounds")
            # The next decode can overwrite the buffer only after frame() returns.
            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=offset).copy()
        return arrays

    def close(self):
        self.mapping.close()
