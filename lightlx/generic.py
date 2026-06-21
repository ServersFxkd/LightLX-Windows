# Generic out-of-core engine: stream ANY mlx-lm-supported model that's too big
# for RAM but fits on disk. Reuses mlx-lm's exact model definition + forward pass
# (so the math is correct for free) and only injects streaming: each decoder layer
# is wrapped so its weights are paged in from disk on demand, run, then freed.
#
# Key enabler: MLX is lazy — Model(args) creates lazy zero/random params that are
# never materialized, so instantiating a model far bigger than RAM doesn't OOM.
# We then page real weights into ONE shared layer module, one layer at a time.
#
# v0 scope: standard dense decoder models (llama / qwen2 / mistral / gemma / phi /
# ... — the bulk of HF). MoE archs that stack experts at load (mixtral, deepseek,
# glm_moe_dsa) need expert-aware streaming; GLM-5.2 has its own engine (model.py).

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.utils import _get_classes

from .weights import StreamingWeights


# expert projection naming per MoE family (file-level per-expert tensors -> gate/up/down)
_EXPERT_PROJS = [
    {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},  # qwen2/3_moe, olmoe, deepseek, granite…
    {"gate": "w1", "up": "w3", "down": "w2"},                       # mixtral / phixtral
]


class StreamingSwitchGLU(nn.Module):
    """Drop-in replacement for mlx-lm's SwitchGLU that holds NO expert weights.
    On each call it reads ONLY the routed (top-k) experts for the current layer
    from disk, stacks them, remaps the indices, and runs SwitchGLU's exact math.
    This is the MoE 'scalpel': per decode token we touch k experts, not all N."""

    def __init__(self, parent, input_dims, hidden_dims, projs):
        super().__init__()
        self.parent = parent          # GenericStreamingModel: gives .w and ._cur_layer
        self.projs = projs            # {'gate':..., 'up':..., 'down':...} file tensor names
        # reusable compute module; its 3 weights are overwritten with the routed subset each call
        self._compute = SwitchGLU(input_dims, hidden_dims, 1, bias=False)

    def __call__(self, x, indices):
        i = self.parent._cur_layer
        w = self.parent.w
        flat = np.array(indices).reshape(-1).tolist()       # forces router eval; tells us which experts
        uniq = sorted(set(int(e) for e in flat))
        remap = {e: k for k, e in enumerate(uniq)}
        pre = f"model.layers.{i}.mlp.experts."

        def stack(proj):
            return mx.stack([w.read(f"{pre}{e}.{proj}.weight") for e in uniq])

        self._compute.gate_proj.update({"weight": stack(self.projs["gate"])})
        self._compute.up_proj.update({"weight": stack(self.projs["up"])})
        self._compute.down_proj.update({"weight": stack(self.projs["down"])})
        local = mx.array([remap[int(e)] for e in flat]).reshape(indices.shape).astype(indices.dtype)
        return self._compute(x, local)


class _StreamLayer:
    """Stands in for one decoder layer. On call, pages that layer's weights into a
    shared proto module, runs the real forward, returns. Proxies attribute reads
    (e.g. `use_sliding`) so the parent model's layer loop works unchanged."""

    def __init__(self, proto, idx, parent, use_sliding=False):
        self.proto = proto
        self.idx = idx
        self.parent = parent
        self.use_sliding = use_sliding

    def __call__(self, *args, **kwargs):
        self.parent._page_in(self.idx)
        out = self.proto(*args, **kwargs)
        # Force THIS layer to compute now, so its weights can be freed before the next
        # layer loads. Without this, mlx-lm builds the whole forward lazily and keeps
        # every layer's weights alive at once → the full model is forced into RAM → swap.
        mx.eval(out, *(a for a in args if isinstance(a, mx.array)))
        self.parent._tick()
        return out

    def __getattr__(self, name):  # defer unknown attrs to the shared proto module
        return getattr(object.__getattribute__(self, "proto"), name)


class GenericStreamingModel:
    def __init__(self, model_dir, verbose=True, max_layers=None):
        self.dir = model_dir
        self.verbose = verbose
        self.streaming = True
        self._on_layer = None
        self._t0 = 0.0
        self._done = 0
        self.w = StreamingWeights(model_dir)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.arch = cfg.get("model_type", "?")
        Model, ModelArgs = _get_classes(cfg)
        self.args = ModelArgs.from_dict(cfg)
        self.model = Model(self.args)  # lazy params -> instantiating a >RAM model is cheap

        inner = getattr(self.model, "model", self.model)
        if not hasattr(inner, "layers"):
            raise RuntimeError(f"arch '{self.arch}': no model.model.layers; not supported by v0 generic engine")
        self.inner = inner
        orig = inner.layers
        n = len(orig)
        self.proto = orig[0]  # reused compute module; weights paged in per layer
        self._cur_layer = 0

        # --- MoE detection: a layer is MoE if its mlp carries a SwitchGLU (switch_mlp) ---
        def _is_moe(layer):
            mlp = getattr(layer, "mlp", None)
            return mlp is not None and hasattr(mlp, "switch_mlp")
        moe_flags = [_is_moe(l) for l in orig]
        self.is_moe = len(moe_flags) > 0 and all(moe_flags)
        if any(moe_flags) and not all(moe_flags):
            raise RuntimeError(
                f"arch '{self.arch}': mixes dense and MoE layers — generic v0 streams uniform-MoE "
                f"models only (OLMoE, Mixtral, Qwen3-MoE, …). GLM-5.2 has its own engine.")

        self._expert_projs = None
        if self.is_moe:
            # discover per-expert file tensor naming (gate/up/down vs mixtral w1/w2/w3)
            sample = next(int(nm.split(".")[2]) for nm in self.w.weight_map if ".mlp.experts." in nm)
            spre = f"model.layers.{sample}.mlp.experts.0."
            present = {nm[len(spre):-len(".weight")] for nm in self.w.weight_map
                       if nm.startswith(spre) and nm.endswith(".weight")}
            self._expert_projs = next((c for c in _EXPERT_PROJS if set(c.values()) <= present), None)
            if self._expert_projs is None:
                raise RuntimeError(f"arch '{self.arch}': unrecognized expert projections {present}")
            # swap the lazy SwitchGLU for the streaming scalpel (reads only top-k experts/token)
            sg = self.proto.mlp.switch_mlp
            in_dims, hid_dims = sg.gate_proj.weight.shape[2], sg.gate_proj.weight.shape[1]
            self.proto.mlp.switch_mlp = StreamingSwitchGLU(self, in_dims, hid_dims, self._expert_projs)

        # precompute per-layer tensor name lists (avoid scanning the full weight_map each layer).
        # For MoE, EXCLUDE the routed-expert tensors — the scalpel pages them on demand.
        self._lnames = {}
        for nm in self.w.weight_map:
            if not nm.startswith("model.layers.") or "rotary_emb.inv_freq" in nm:
                continue
            if self.is_moe and ".mlp.experts." in nm:
                continue
            i = int(nm.split(".")[2])
            self._lnames.setdefault(i, []).append(nm)

        if verbose:
            moe = ""
            if self.is_moe:
                ne = getattr(self.args, "num_experts", getattr(self.args, "num_local_experts", "?"))
                tk = getattr(self.args, "num_experts_per_tok", "?")
                moe = f", MoE scalpel: {tk}/{ne} experts per token"
            print(f"streaming '{self.arch}': {n} layers, hidden={getattr(self.args,'hidden_size','?')}{moe}", flush=True)
        self._load_resident()

        wrappers = [_StreamLayer(self.proto, i, self, getattr(orig[i], "use_sliding", False))
                    for i in range(n)]
        if max_layers:  # debug: run only the first N layers (output not meaningful)
            wrappers = wrappers[:max_layers]
        inner.layers = wrappers
        self.n = len(wrappers)

    def _load_resident(self):
        # embeddings + final norm (+ lm_head only if present & untied) stay resident
        if self.w.has("model.embed_tokens.weight"):
            self.inner.embed_tokens.update({"weight": self.w.get("model.embed_tokens.weight")})
        if self.w.has("model.norm.weight"):
            self.inner.norm.update({"weight": self.w.get("model.norm.weight")})
        if self.w.has("lm_head.weight") and hasattr(self.model, "lm_head"):
            self.model.lm_head.update({"weight": self.w.get("lm_head.weight")})
        mx.eval(self.inner.embed_tokens.parameters(), self.inner.norm.parameters())
        if self.verbose:
            print(f"  resident set: {self.w.bytes_read/1e9:.2f} GB", flush=True)

    def _page_in(self, i):
        self._cur_layer = i  # tells StreamingSwitchGLU which layer's experts to fetch
        mx.clear_cache()  # free the previous layer's weights
        pre = f"model.layers.{i}."
        flat = {nm[len(pre):]: self.w.get(nm) for nm in self._lnames.get(i, [])}  # pread, materialized
        self.proto.update(tree_unflatten(list(flat.items())))
        mx.eval(self.proto.parameters())

    def _tick(self):
        self._done += 1
        if self._on_layer is not None:
            self._on_layer(self._done, self.n, self.w.bytes_read, time.time() - self._t0)

    def __call__(self, inputs, cache, on_layer=None):
        self._on_layer = on_layer
        self._t0 = time.time()
        self._done = 0
        return self.model(inputs, cache)

    def make_cache(self):
        return make_prompt_cache(self.model)


class _Bytes:
    def __init__(self):
        self.bytes_read = 0


class ResidentModel:
    """Fast path: the model FITS in RAM, so load it fully resident and run on the
    GPU at full speed — exactly like LM Studio / Ollama. No streaming, no overhead.
    Exposes the same interface the CLI's generate loop expects."""

    def __init__(self, model_dir, verbose=True):
        from mlx_lm import load as _load
        if verbose:
            print("model fits in memory — loading fully resident (GPU, fast)...", flush=True)
        self.model, self._tok = _load(model_dir)  # full in-memory load + eval
        self.streaming = False
        self.w = _Bytes()  # bytes_read stays 0: nothing streams during generation
        if verbose:
            print("  loaded resident.", flush=True)

    def __call__(self, inputs, cache, on_layer=None):
        return self.model(inputs, cache)  # native forward, whole model on GPU

    def make_cache(self):
        return make_prompt_cache(self.model)
