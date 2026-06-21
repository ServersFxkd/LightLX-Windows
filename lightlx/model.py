# Streaming GLM-5.2 (glm_moe_dsa) forward pass — the LightLX engine, v1.
#
# Reuses mlx-lm's exact GLM-5.2 / DeepSeek-V3.2 math (MLA attention, the
# noaux_tc sigmoid router, RoPE, RMSNorm, MultiLinear absorption) but never
# instantiates the full ~1.5 TB model. It pages each decoder layer through ONE
# reusable attention module and loads only the top-8 routed experts per token.
#
# v1 adds the "sliding window" residency layer:
#   * PINNED (resident, read once at startup, never re-streamed): everything
#     touched on every token that fits in RAM -- embeddings, final norm, lm_head,
#     all router gates, all layernorms, all 75 shared experts, the 3 dense MLPs,
#     and optionally the first N layers' attention.
#   * CACHED (bounded LRU): routed experts, so repeats across tokens skip disk.
#   * STREAMED (cold): everything else (most routed experts, unpinned attention).
#
# At full BF16 on 24 GB the pinned + cache set is small relative to the 724 GB of
# routed experts, so the win is modest (~1.2-1.3x) -- but the architecture scales
# directly with RAM and storage speed (and with lower-precision weights).
#
# v1 scope: full BF16, no quantization, dense MLA (DSA indexer bypassed -> exact
# for context <= index_topk=2048). Greedy decoding.

import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v32 import group_expert_select
from mlx_lm.models.glm_moe_dsa import ModelArgs
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.rope_utils import initialize_rope

from .cache import LayeredExpertCache
from .weights import Prefetcher, StreamingWeights


def _try(fns, nbytes: int) -> bool:
    for fn in fns:
        if fn is not None:
            try:
                fn(int(nbytes))
                return True
            except Exception:
                pass
    return False


def _set_wired_limit(nbytes: int) -> bool:
    """Best-effort wire the resident set so macOS can't swap it."""
    return _try([getattr(mx, "set_wired_limit", None),
                 getattr(getattr(mx, "metal", None), "set_wired_limit", None)], nbytes)


def _set_cache_limit(nbytes: int) -> bool:
    """Cap MLX's buffer reuse pool so it returns freed memory to the OS.
    Without this MLX hoards GBs of freed buffers -> memory pressure -> UI stutter."""
    return _try([getattr(mx, "set_cache_limit", None),
                 getattr(getattr(mx, "metal", None), "set_cache_limit", None)], nbytes)


_GS = 64  # quantization group size


def _quant(W, bits):
    wq, s, b = mx.quantize(W, group_size=_GS, bits=bits)
    return (wq, s, b)


def _qmm(x, q, bits):
    """x @ W.T using a quantized W = (weight, scales, biases)."""
    wq, s, b = q
    return mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True, group_size=_GS, bits=bits)


class StreamAttention(nn.Module):
    """MLA attention, indexer-free (valid for ctx <= index_topk). One instance
    is reused for every layer; real weights are paged in via .update()."""

    def __init__(self, a: ModelArgs):
        super().__init__()
        self.num_heads = a.num_attention_heads
        self.qk_rope_head_dim = a.qk_rope_head_dim
        self.qk_nope_head_dim = a.qk_nope_head_dim
        self.kv_lora_rank = a.kv_lora_rank
        self.v_head_dim = a.v_head_dim
        self.q_head_dim = a.qk_nope_head_dim + a.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(a.hidden_size, a.q_lora_rank, bias=a.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(a.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(a.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            a.hidden_size, a.kv_lora_rank + a.qk_rope_head_dim, bias=a.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(a.kv_lora_rank, eps=1e-6)
        self.embed_q = MultiLinear(a.qk_nope_head_dim, a.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(a.kv_lora_rank, a.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * a.v_head_dim, a.hidden_size, bias=a.attention_bias)
        self.rope = initialize_rope(
            dims=a.qk_rope_head_dim, base=a.rope_theta, traditional=True,
            max_position_embeddings=a.max_position_embeddings, scaling_config=a.rope_scaling,
        )

    def __call__(self, x, mask, cache):
        B, L, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        out = mx.fast.scaled_dot_product_attention(q_nope, k, v, scale=self.scale, mask=pe_scores)
        if L == 1:
            out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class StreamingGLM:
    def __init__(self, model_dir, args: ModelArgs, verbose=True,
                 expert_cache_gb: float = 0.0, pin_attn_layers: int = 0,
                 wired_gb: float | None = None, skeleton_bits=None, prefetch: bool = False):
        self.dir = model_dir
        self.a = args
        self.streaming = True
        self.w = StreamingWeights(model_dir)
        self.prefetch = Prefetcher(self.w) if prefetch else None  # async page-cache warming
        self.gather = False  # gather/scatter MoE — benchmarked SLOWER (0.76x); see notes. off by default.
        self.verbose = verbose
        self.qbits = skeleton_bits           # None = full BF16; 4 = quantize skeleton on load
        self.attn = StreamAttention(args)
        if self.qbits:
            nn.quantize(self.attn, group_size=_GS, bits=self.qbits)  # quantized module structure
        moe_layers = [l for l in range(args.num_hidden_layers) if l >= args.first_k_dense_replace]
        self.expert_cache = LayeredExpertCache(int(expert_cache_gb * 1e9), moe_layers)
        self.pin_attn_layers = args.num_hidden_layers if self.qbits else min(pin_attn_layers, args.num_hidden_layers)

        if verbose:
            mode = f"{self.qbits}-bit skeleton, BF16 experts" if self.qbits else "full BF16"
            action = "reading + quantizing" if self.qbits else "reading"
            print(f"pinning resident skeleton ({mode}) — {action} from your weights...", flush=True)

        # embeddings (lookup) + final norm stay BF16; lm_head quantized when self.qbits
        self.embed_w = self.w.fetch(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
        self.norm_w = self.w.fetch(["model.norm.weight"])["model.norm.weight"]
        lmh = self.w.fetch(["lm_head.weight"])["lm_head.weight"]
        self.lm_head_q = _quant(lmh, self.qbits) if self.qbits else None
        self.lm_head_w = None if self.qbits else lmh

        self.res_norm, self.res_gate = {}, {}
        self.res_shared, self.res_dense, self.res_attn = {}, {}, {}        # BF16 path
        self.res_shared_q, self.res_dense_q, self.res_attn_q = {}, {}, {}  # quantized path
        for l in range(args.num_hidden_layers):
            self.res_norm[l] = self.w.fetch([
                f"model.layers.{l}.input_layernorm.weight",
                f"model.layers.{l}.post_attention_layernorm.weight",
            ])
            if l < args.first_k_dense_replace:                       # dense MLP layer
                if self.qbits:
                    self.res_dense_q[l] = self._load_mlp_q(f"model.layers.{l}.mlp")
                else:
                    self.res_dense[l] = self._mlp_names(f"model.layers.{l}.mlp")
            else:                                                    # MoE layer
                p = f"model.layers.{l}.mlp"
                self.res_gate[l] = self.w.fetch([f"{p}.gate.weight", f"{p}.gate.e_score_correction_bias"])
                if self.qbits:
                    self.res_shared_q[l] = self._load_mlp_q(f"{p}.shared_experts")
                else:
                    self.res_shared[l] = self._mlp_names(f"{p}.shared_experts")
            if self.qbits:                                           # attention -> resident, quantized
                self.res_attn_q[l] = self._attn_quant(l, args)
            elif l < self.pin_attn_layers:
                self.res_attn[l] = self._attn_flat(l)
            if verbose and l % 8 == 0:
                print(f"\r  through layer {l:2d}/{args.num_hidden_layers} ({self.w.bytes_read/1e9:.0f} GB read)",
                      end="", flush=True)
        if verbose:
            print(f"\r  skeleton resident ({mode}); read {self.w.bytes_read/1e9:.0f} GB BF16 from disk"
                  + " " * 12)

        ram = _total_ram_gb()
        resident_gb = 13.0 if self.qbits else self.w.bytes_read / 1e9
        if wired_gb is None:
            # Wire ~the resident set, capped well below RAM so the OS/UI stays responsive.
            wired_gb = min(resident_gb + 1.0, 0.6 * ram)
        _set_wired_limit(wired_gb * 1e9)
        _set_cache_limit(int(1.5e9))  # keep MLX from hoarding freed buffers -> smoother UI
        if verbose:
            print(f"  wired ~{wired_gb:.0f} GB of {ram:.0f} GB (leaving ~{ram - wired_gb:.0f} GB for the OS)\n")

    def _mlp_names(self, prefix):
        return self.w.fetch([f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight",
                             f"{prefix}.down_proj.weight"])

    def _load_mlp_q(self, prefix):
        """Load an MLP's 3 matrices from BF16 and quantize them to self.qbits (resident)."""
        t = self.w.fetch([f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight", f"{prefix}.down_proj.weight"])
        return {"gate": _quant(t[f"{prefix}.gate_proj.weight"], self.qbits),
                "up":   _quant(t[f"{prefix}.up_proj.weight"], self.qbits),
                "down": _quant(t[f"{prefix}.down_proj.weight"], self.qbits)}

    def _q_swiglu(self, q, x):
        """SwiGLU MLP with quantized weights q={'gate','up','down'}."""
        return _qmm(swiglu(_qmm(x, q["gate"], self.qbits), _qmm(x, q["up"], self.qbits)), q["down"], self.qbits)

    def _attn_quant(self, l, args):
        """Load a layer's BF16 attention, absorb kv_b, quantize, return resident params."""
        tmp = StreamAttention(args)
        tmp.update(tree_unflatten(list(self._attn_flat(l).items())))
        nn.quantize(tmp, group_size=_GS, bits=self.qbits)
        return tmp.parameters()

    def _attn_flat(self, l: int) -> dict:
        """Fetch + absorb one layer's attention weights into a flat update dict."""
        p = f"model.layers.{l}.self_attn"
        t = self.w.fetch([
            f"{p}.q_a_proj.weight", f"{p}.q_a_layernorm.weight", f"{p}.q_b_proj.weight",
            f"{p}.kv_a_proj_with_mqa.weight", f"{p}.kv_a_layernorm.weight",
            f"{p}.kv_b_proj.weight", f"{p}.o_proj.weight",
        ])
        a = self.a
        head_dim = a.qk_nope_head_dim + a.v_head_dim
        v = t[f"{p}.kv_b_proj.weight"].reshape(a.num_attention_heads, head_dim, -1)
        wk = mx.contiguous(v[:, : a.qk_nope_head_dim, :].swapaxes(-1, -2))
        wv = mx.contiguous(v[:, a.qk_nope_head_dim :, :])
        return {
            "q_a_proj.weight": t[f"{p}.q_a_proj.weight"],
            "q_a_layernorm.weight": t[f"{p}.q_a_layernorm.weight"],
            "q_b_proj.weight": t[f"{p}.q_b_proj.weight"],
            "kv_a_proj_with_mqa.weight": t[f"{p}.kv_a_proj_with_mqa.weight"],
            "kv_a_layernorm.weight": t[f"{p}.kv_a_layernorm.weight"],
            "embed_q.weight": wk, "unembed_out.weight": wv,
            "o_proj.weight": t[f"{p}.o_proj.weight"],
        }

    def _run_attention(self, l, xn, mask, cache):
        if self.qbits:                                 # resident, quantized — no disk read
            self.attn.update(self.res_attn_q[l])
        else:
            flat = self.res_attn.get(l) or self._attn_flat(l)  # pinned BF16 or streamed
            self.attn.update(tree_unflatten(list(flat.items())))
        return self.attn(xn, mask, cache)

    @staticmethod
    def _swiglu_mlp(t, prefix, x):
        h = swiglu(x @ t[f"{prefix}.gate_proj.weight"].T, x @ t[f"{prefix}.up_proj.weight"].T)
        return h @ t[f"{prefix}.down_proj.weight"].T

    def _mlp_moe(self, l, x):
        a = self.a
        p = f"model.layers.{l}.mlp"
        g = self.res_gate[l]
        gates = x @ g[f"{p}.gate.weight"].T
        inds, scores = group_expert_select(
            gates, g[f"{p}.gate.e_score_correction_bias"], a.num_experts_per_tok,
            a.n_group, a.topk_group, a.routed_scaling_factor, a.norm_topk_prob,
        )
        mx.eval(inds, scores)
        B, L, H = x.shape
        N = B * L
        # Build per-expert token index + weight lists on CPU from the routing we already
        # evaluated (no extra GPU syncs). Each token routes to K distinct experts.
        inds_l = inds.reshape(N, -1).tolist()
        scores_l = scores.reshape(N, -1).tolist()
        tok, wt = {}, {}
        for n in range(N):
            for e_id, sc in zip(inds_l[n], scores_l[n]):
                tok.setdefault(e_id, []).append(n)
                wt.setdefault(e_id, []).append(sc)
        union = sorted(tok.keys())

        if self.prefetch is not None:  # warm this layer's experts into page cache while we compute
            names = []
            for e in union:
                pe = f"{p}.experts.{e}"
                names += [f"{pe}.gate_proj.weight", f"{pe}.up_proj.weight", f"{pe}.down_proj.weight"]
            self.prefetch.submit(names)

        def expert_weights(e):
            pe = f"{p}.experts.{e}"
            t = self.expert_cache.get(l, e)
            if t is None:                                   # miss -> stream (always full BF16) + cache
                t = self.w.fetch([f"{pe}.gate_proj.weight", f"{pe}.up_proj.weight", f"{pe}.down_proj.weight"])
                self.expert_cache.put(l, e, t)
            return t, pe

        if self.gather:
            # GATHER/SCATTER: compute each expert ONLY on the tokens routed to it.
            # (prefill computes ~K*N expert-rows instead of |union|*N — no waste, exact.)
            xf = x.reshape(N, H)
            out = mx.zeros((N, H), dtype=mx.float32)
            for e in union:
                t, pe = expert_weights(e)
                idx = mx.array(tok[e])
                w = mx.array(wt[e]).astype(mx.float32)
                ye = self._swiglu_mlp(t, pe, xf[idx])       # [n_e, H] — only routed rows
                out = out.at[idx].add(ye.astype(mx.float32) * w[:, None])
            out = out.reshape(B, L, H).astype(x.dtype)
        else:
            # ORIGINAL: compute every expert over ALL positions, then mask (wasteful in prefill).
            out = mx.zeros(x.shape, dtype=mx.float32)
            for e in union:
                t, pe = expert_weights(e)
                ye = self._swiglu_mlp(t, pe, x)
                wgt = (scores * (inds == e)).sum(axis=-1)
                out = out + ye.astype(mx.float32) * wgt[..., None]
            out = out.astype(x.dtype)

        shared = (self._q_swiglu(self.res_shared_q[l], x) if self.qbits
                  else self._swiglu_mlp(self.res_shared[l], f"{p}.shared_experts", x))
        return out + shared

    def __call__(self, input_ids, cache, on_layer=None):
        a = self.a
        h = self.embed_w[input_ids]
        mask = create_attention_mask(h, cache[0], return_array=True)
        t0 = time.time()
        for l in range(a.num_hidden_layers):
            ln_in, ln_post = self.res_norm[l].values()
            h = h + self._run_attention(l, mx.fast.rms_norm(h, ln_in, a.rms_norm_eps), mask, cache[l])
            xn = mx.fast.rms_norm(h, ln_post, a.rms_norm_eps)
            if l < a.first_k_dense_replace:
                dense = (self._q_swiglu(self.res_dense_q[l], xn) if self.qbits
                         else self._swiglu_mlp(self.res_dense[l], f"model.layers.{l}.mlp", xn))
                h = h + dense
            else:
                h = h + self._mlp_moe(l, xn)
            mx.eval(h, cache[l].state)
            mx.clear_cache()  # frees transient buffers; pinned + cached arrays survive (referenced)
            if on_layer is not None:
                on_layer(l + 1, a.num_hidden_layers, self.w.bytes_read, time.time() - t0)
        h = mx.fast.rms_norm(h, self.norm_w, a.rms_norm_eps)
        if self.qbits:
            return _qmm(h[:, -1:, :], self.lm_head_q, self.qbits)
        return h[:, -1:, :] @ self.lm_head_w.T

    def make_cache(self):
        return [KVCache() for _ in range(self.a.num_hidden_layers)]


def _total_ram_gb() -> float:
    try:
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9
    except Exception:
        return 24.0
