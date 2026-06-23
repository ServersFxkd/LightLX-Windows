# Generic out-of-core engine: PyTorch + HuggingFace version.

import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from .weights import StreamingWeights

class _StreamLayer(torch.nn.Module):
    """Wraps a single decoder layer. Pages weights in from disk, runs forward, frees them."""
    def __init__(self, proto, idx, parent):
        super().__init__()
        self.proto = proto
        self.idx = idx
        self.parent = parent

    def forward(self, *args, **kwargs):
        self.parent._page_in(self.idx)
        out = self.proto(*args, **kwargs)
        return out


class GenericStreamingModel:
    def __init__(self, model_dir, verbose=True, max_layers=None, device="cpu"):
        self.dir = model_dir
        self.verbose = verbose
        self.streaming = True
        self._on_layer = None
        self._t0 = 0.0
        self._done = 0
        self.device = device
        
        self.w = StreamingWeights(model_dir)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.arch = config.model_type

        # Instantiate the model with NO weights (uses accelerate's meta device)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        self.model.to(self.device)

        inner = getattr(self.model, "model", self.model)
        if not hasattr(inner, "layers") and not hasattr(inner, "h"):
            raise RuntimeError(f"arch '{self.arch}': not supported by v0 generic engine.")
        
        orig = getattr(inner, "layers", getattr(inner, "h", None))
        n = len(orig)
        self.proto = orig[0]
        self._cur_layer = 0

        proto_keys = list(self.proto.state_dict().keys())
        self.layer_prefix = ""
        for pk in proto_keys:
            matches = [sk for sk in self.w.weight_map if sk.endswith(pk)]
            if matches:
                self.layer_prefix = matches[0][:-len(pk)]
                break
        
        if not self.layer_prefix:
            raise RuntimeError("Could not map layer weights to safetensors keys.")

        self._lnames = {}
        for nm in self.w.weight_map:
            if self.layer_prefix.replace("0", "") in nm:
                try:
                    i = int(nm.split(".")[2] if "model.layers." in nm else nm.split(".")[1])
                    self._lnames.setdefault(i, []).append(nm)
                except (IndexError, ValueError):
                    continue

        if verbose:
            print(f"streaming '{self.arch}': {n} layers", flush=True)
        
        self._load_resident()

        wrappers = [_StreamLayer(self.proto, i, self) for i in range(n)]
        if max_layers:
            wrappers = wrappers[:max_layers]
        
        if hasattr(inner, "layers"):
            inner.layers = torch.nn.ModuleList(wrappers)
        elif hasattr(inner, "h"):
            inner.h = torch.nn.ModuleList(wrappers)
            
        self.n = len(wrappers)

    def _load_resident(self):
        for name, param in self.model.named_parameters():
            if not name.startswith(self.layer_prefix.replace("0", "")):
                full_name = name
                if self.w.has(full_name):
                    tensor = self.w.read(full_name).to(self.device)
                    param.data = tensor

    def _page_in(self, i):
        self._cur_layer = i
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        state_dict = {}
        prefix = self.layer_prefix.replace("0", str(i))
        for name, _ in self.proto.state_dict().items():
            full_name = prefix + name
            if self.w.has(full_name):
                state_dict[name] = self.w.read(full_name).to(self.device)
                
        self.proto.load_state_dict(state_dict, strict=False)

    def _tick(self):
        self._done += 1
        if self._on_layer is not None:
            self._on_layer(self._done, self.n, self.w.bytes_read, time.time() - self._t0)

    def __call__(self, inputs, cache=None, on_layer=None):
        self._on_layer = on_layer
        self._t0 = time.time()
        self._done = 0
        
        with torch.no_grad():
            out = self.model(inputs, past_key_values=cache, use_cache=True)
            self._tick() # Report layer progress
            
        # Return both logits and the updated KV cache
        return out.logits, out.past_key_values

    def make_cache(self):
        return None  # None is valid for first call; HF returns cache in output


class _Bytes:
    def __init__(self):
        self.bytes_read = 0

class ResidentModel:
    def __init__(self, model_dir, verbose=True, device="cpu"):
        if verbose:
            print("model fits in memory — loading fully resident (fast)...", flush=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )
        self.streaming = False
        self.w = _Bytes()
        if verbose:
            print("  loaded resident.", flush=True)

    def __call__(self, inputs, cache=None, on_layer=None):
        with torch.no_grad():
            out = self.model(inputs, past_key_values=cache, use_cache=True)
        # Return both logits and the updated KV cache
        return out.logits, out.past_key_values

    def make_cache(self):
        return None  # None is valid for first call; HF returns cache in output