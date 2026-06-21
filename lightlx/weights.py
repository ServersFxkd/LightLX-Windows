# Streaming weight store: reads individual tensors on demand from the original
# safetensors shards via direct os.pread (NOT mmap).
#
# Why not mmap (mx.load): mmap'd shard pages accumulate in the OS page cache, and
# on a model bigger than RAM macOS swaps out process memory to keep them ->
# catastrophic thrash. Direct pread pages are reclaimable, so there's no swap.
# This single change took a >RAM dense model from ~170 s/token (swap-bound) to
# ~11-13 s/token (disk-bound), with bit-identical output.
#
# read() is split into _pread (raw bytes, no MLX -> thread-safe) + _to_mx (build
# the mx.array, main thread only). That split keeps the raw I/O safe to call from
# the Prefetcher's worker threads (GLM page-cache warming). NB: overlapping the
# next layer's read with the current layer's compute was tried for single-stream
# decode and did NOT help (decode compute is µs/layer — the read is the critical
# path), so the generic engine reads synchronously.

import glob
import json
import os
import queue
import struct
import threading

import mlx.core as mx
import numpy as np

try:
    import ml_dtypes
    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    _BF16 = None

# safetensors dtype string -> numpy dtype (bf16 via ml_dtypes)
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
                self.weight_map = json.load(f)["weight_map"]  # tensor name -> shard file
        else:
            # single-file or sharded-without-index: build the map from each file's header
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
        # Pre-read every shard's safetensors header once, so _offsets is a pure dict
        # lookup -> safe to call from prefetch worker threads (no shared-state races).
        self._headers: dict = {}  # shard path -> (header_json, data_base_offset)
        for shard in set(self.weight_map.values()):
            path = os.path.join(self.dir, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[path] = (json.loads(f.read(n)), 8 + n)

    def _offsets(self, name: str):
        """Resolve a tensor to (shard_path, file_offset, nbytes) via the safetensors header."""
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

    def warm(self, name: str):
        """Read a tensor's raw bytes into the OS page cache (thread-safe, no MLX)."""
        try:
            path, off, ln = self._offsets(name)
            fd = os.open(path, os.O_RDONLY)
            try:
                got = 0
                while got < ln:
                    chunk = os.pread(fd, min(ln - got, 8 << 20), off + got)
                    if not chunk:
                        break
                    got += len(chunk)
            finally:
                os.close(fd)
        except Exception:
            pass  # prefetch is best-effort; a miss just means the main read hits disk

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _pread(self, path: str, off: int, ln: int) -> bytes:
        """Raw chunked pread of a byte range. Pure I/O, no MLX -> thread-safe."""
        fd = os.open(path, os.O_RDONLY)
        try:
            chunks, got = [], 0
            while got < ln:
                c = os.pread(fd, min(ln - got, 16 << 20), off + got)
                if not c:
                    break
                chunks.append(c)
                got += len(c)
        finally:
            os.close(fd)
        return b"".join(chunks)

    def _to_mx(self, name: str, buf: bytes) -> mx.array:
        """Build a materialized mx.array from a tensor's raw bytes. Main thread only."""
        meta = self._headers[os.path.join(self.dir, self.weight_map[name])][0][name]
        dtype = _NP_DTYPE.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']} for {name}")
        arr = np.frombuffer(buf, dtype=dtype)
        if meta["shape"]:
            arr = arr.reshape(meta["shape"])
        return mx.array(arr)  # copies into MLX memory; the byte buffer is then freed

    def read(self, name: str) -> mx.array:
        """Read one tensor synchronously via pread -> materialized mx.array."""
        path, off, ln = self._offsets(name)
        buf = self._pread(path, off, ln)
        self.tensors_read += 1
        self.bytes_read += ln
        return self._to_mx(name, buf)

    def get(self, name: str) -> mx.array:
        return self.read(name)

    def fetch(self, names, eval_now: bool = True) -> dict:
        out = {n: self.read(n) for n in names}
        if eval_now:
            mx.eval(list(out.values()))
        return out


class Prefetcher:
    """Background threads that warm upcoming tensors into the OS page cache, so the
    main thread's mx.load reads from RAM instead of disk. Pure file I/O on workers
    (no MLX calls) -> thread-safe. Keeps the disk fed during compute/eval gaps."""

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
                break  # already plenty queued; don't block the compute thread
