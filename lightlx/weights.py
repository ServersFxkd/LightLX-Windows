# Streaming weight store: PyTorch + Windows compatible.

import glob
import json
import os
import queue
import struct
import threading

import torch
import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None

_NP_DTYPE = {
    "BF16": _BF16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BOOL": np.bool_,
}

class StreamingWeights:
    def __init__(self, model_dir: str, shard_cache: int = 12):
        self.dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                fn = os.path.basename(path)
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
                for name in hdr:
                    if name != "__metadata__":
                        self.weight_map[name] = fn
            if not self.weight_map:
                raise FileNotFoundError(f"no safetensors weights found in {model_dir}")
        
        self.bytes_read = 0
        self.tensors_read = 0
        self._headers: dict = {}
        for shard in set(self.weight_map.values()):
            path = os.path.join(self.dir, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[path] = (json.loads(f.read(n)), 8 + n)

    def _offsets(self, name: str):
        path = os.path.join(self.dir, self.weight_map[name])
        hdr = self._headers.get(path)
        if hdr is None:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = (json.loads(f.read(n)), 8 + n)
            self._headers[path] = hdr
        meta, base = hdr
        s, e = meta[name]["data_offsets"]
        return path, base + s, e - s

    def _safe_pread(self, fd, length, offset):
        """Cross-platform pread. Windows doesn't have os.pread before Python 3.13."""
        try:
            return os.pread(fd, length, offset)
        except AttributeError:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, length)

    def warm(self, name: str):
        try:
            path, off, ln = self._offsets(name)
            fd = os.open(path, os.O_RDONLY)
            try:
                got = 0
                while got < ln:
                    chunk = self._safe_pread(fd, min(ln - got, 8 << 20), off + got)
                    if not chunk:
                        break
                    got += len(chunk)
            finally:
                os.close(fd)
        except Exception:
            pass

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _pread(self, path: str, off: int, ln: int) -> bytes:
        fd = os.open(path, os.O_RDONLY)
        try:
            chunks, got = [], 0
            while got < ln:
                c = self._safe_pread(fd, min(ln - got, 16 << 20), off + got)
                if not c:
                    break
                chunks.append(c)
                got += len(c)
        finally:
            os.close(fd)
        return b"".join(chunks)

    def _to_torch(self, name: str, buf: bytes) -> torch.Tensor:
        meta = self._headers[os.path.join(self.dir, self.weight_map[name])][0][name]
        dtype_str = meta["dtype"]
        
        if dtype_str == "BF16":
            arr = np.frombuffer(buf, dtype=np.int16)
            t = torch.from_numpy(arr)
            t = t.view(torch.bfloat16)
        else:
            dtype = _NP_DTYPE.get(dtype_str)
            if dtype is None:
                raise ValueError(f"unsupported safetensors dtype {dtype_str} for {name}")
            arr = np.frombuffer(buf, dtype=dtype)
            t = torch.from_numpy(arr)
            
        if meta["shape"]:
            t = t.reshape(meta["shape"])
        return t.contiguous()

    def read(self, name: str) -> torch.Tensor:
        path, off, ln = self._offsets(name)
        buf = self._pread(path, off, ln)
        self.tensors_read += 1
        self.bytes_read += ln
        return self._to_torch(name, buf)

    def get(self, name: str) -> torch.Tensor:
        return self.read(name)

    def fetch(self, names) -> dict:
        return {n: self.read(n) for n in names}


class Prefetcher:
    def __init__(self, weights: StreamingWeights, workers: int = 2, depth: int = 64):
        self.w = weights
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def _run(self):
        while True:
            name = self.q.get()
            if name is None:
                return
            self.w.warm(name)

    def submit(self, names):
        for n in names:
            try:
                self.q.put_nowait(n)
            except queue.Full:
                break