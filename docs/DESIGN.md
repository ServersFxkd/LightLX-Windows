# LightLX — Design Notes

How LightLX runs models that don't fit in RAM, why it's built the way it is, and
what we measured along the way. Everything here is from runs on a **MacBook Pro
M4 Pro, 24 GB** unified memory.

---

## The problem: the memory wall

A transformer must read every weight it uses for **every token**. Tools like LM
Studio and Ollama load the whole model into RAM first — so the largest model you
can run is capped by your memory. No Apple Silicon machine, not even a 512 GB Mac
Studio, can hold full-precision GLM-5.2 (~1.5 TB).

But a model that doesn't fit in **RAM** usually *does* fit on **disk**. LightLX
runs it from there: keep a small part resident, stream the rest layer-by-layer as
each token needs it, and free it again. The model never has to fit in memory — only
on the SSD.

The cost is speed, and it's honest physics:

```
tokens/sec  ≈  disk bandwidth  ÷  active bytes per token
```

A dense model reads its *entire* weight set per token; a Mixture-of-Experts model
reads only the handful of experts the router picks. LightLX exploits that.

---

## Three execution modes (auto-selected)

LightLX picks a mode by size, then architecture:

1. **Resident** — the model fits in RAM → load it fully onto the GPU and run at
   native speed, exactly like LM Studio / Ollama. No streaming.
2. **Generic streaming** — too big for RAM, any mlx-lm-supported architecture:
   - **Dense** (Llama, Qwen, Mistral, Gemma, Phi): stream each decoder layer.
   - **MoE** (Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek): stream only the routed
     top-k experts per token — the *scalpel* (see below).
3. **GLM-5.2** — a bespoke MoE-native engine for the 753B flagship, with a
   resident skeleton and the MLA / DSA specifics.

The math always comes from mlx-lm's real forward pass — LightLX replaces only
weight *residency*, never the numerics. Output is **bit-identical to native**
(verified `max|Δ logits| = 0` on both a dense and an MoE model).

---

## How streaming works

MLX is lazy: instantiating `Model(args)` allocates *lazy* zero-parameters that are
never materialized, so building a model far larger than RAM costs almost nothing.
LightLX then:

- loads a small **resident set** (embeddings, final norm, lm_head) once;
- replaces each decoder layer with a wrapper that **pages that layer's weights in
  from disk, runs the real forward, then frees them** before the next layer;
- forces `mx.eval` + `mx.clear_cache()` per layer so peak memory stays bounded.

### Why `pread`, not `mmap` — the swap fix

The first version read tensors with `mx.load`, which **mmaps** the safetensors
shards. On a model larger than RAM this is catastrophic: every mmap'd page lands in
the OS page cache, and macOS then **swaps out process memory** to keep those pages
resident. Measured on a 28 GB dense model (Qwen3-14B) on 24 GB: **~170 s/token**
with **+10 GB of swap** during a single read — pure thrash.

The fix: read each tensor with direct `os.pread` → `np.frombuffer` (bf16 via
`ml_dtypes`) → `mx.array`. `pread` pages are immediately reclaimable, so there's no
swap pressure. Result: **170 → 11 s/token (~15×), zero swap growth, bit-identical
output.** This single change is what makes out-of-core inference viable, and it
holds at **1.5 TB scale** (the GLM-5.2 run showed swap *declining*).

---

## The MoE scalpel

A Mixture-of-Experts layer has many experts (64–256) but the router activates only
a few per token (top-8). So per token you need a small, *data-dependent* slice of
the model — and that's exactly what streaming can exploit.

mlx-lm represents experts with one shared module, `SwitchGLU`, whose
`SwitchLinear.__call__` does `mx.gather_mm(x, weight[indices])`. The key insight:
**if you supply only the routed experts' weights and remap the indices, the result
is identical.** And the HF format stores experts as *separate* per-expert tensors
(`...mlp.experts.{e}.gate_proj.weight`), so they can be read individually.

LightLX swaps in a `StreamingSwitchGLU` that holds **no expert weights**. On each
call it reads only the unique routed experts for the current layer from disk,
stacks them, remaps the indices, and runs `SwitchGLU`'s exact math. Because nearly
every mlx-lm MoE architecture shares `SwitchGLU`, this **one replacement** covers
Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek, and more — verified bit-identical (`Δ=0`) on
OLMoE.

> **Why this matters:** streaming *whole* MoE layers would read every expert
> (~all 256/layer on GLM-5.2). Reading only the routed experts is ~16–32× less I/O
> per token — the difference between a model that finishes a token in minutes and one
> that effectively never does.

v0 supports **uniform-MoE** models (every layer MoE); mixed dense/MoE stacks raise a
clear error.

---

## Performance: the measured truth

Single-stream decode is bound by `seconds/token = active-bytes ÷ disk-speed`. On
24 GB you can keep only a small fraction resident, so the rest streams every token.
We tried several software optimizations and **measured** each one:

| Optimization | Result | Why |
|---|---|---|
| Expert LRU cache | **~0% hit** | MoE routing is too diverse; 24 GB caches almost nothing |
| Async prefetch | **0.99×** | prefill is largely disk-*idle* — not I/O-bound to begin with |
| Gather/scatter MoE | **0.76× (slower)** | overhead-bound; for real prompts ~all experts activate anyway |
| Read/compute overlap | **0.86× (slower)** | decode compute is µs/layer — nothing to hide the read behind |
| Queue-depth (parallel reads) | **~1.25×** | the SSD is near its ceiling for this access pattern |
| `--fast` (4-bit skeleton) | **~1.2×** | the one win — the skeleton stops streaming |

### Where a layer's time actually goes

Per decoder layer (Qwen3-14B, internal SSD), measured:

| stage | time | share |
|---|---|---|
| disk read | ~175 ms | **~87%** |
| copy → mx.array (one CPU core) | ~25 ms | ~12% |
| GPU compute | ~1.5 ms | **<1%** |

This is why the CPU and GPU look idle: **LLM decode is memory-bound, not
compute-bound** (~1 FLOP per byte moved; a GPU wants ~20). Even a *resident* 14B on
this machine would cap around 10 tok/s — limited by RAM bandwidth (273 GB/s), with
the GPU mostly waiting. Streaming just swaps RAM for an SSD ~70–100× slower, turning
that invisible wait into a visible one. The only way to use the GPU is **batching**
many sequences (throughput), which a single interactive chat doesn't have.

### Conclusion

No software trick is a big win on 24 GB + a ~1 GB/s drive. The real levers are
outside the box:

1. **Faster storage** (Thunderbolt-5 NVMe, ~5 GB/s) — the cleanest ~5× at full
   precision; disk bandwidth is 87% of the time and scales linearly.
2. **More RAM / lower precision** — shrink the bytes that stream per token.
3. **Two-tier** — a small resident model for instant replies, the big one on demand.

---

## Validation

- **Dense, bit-identical:** generic engine vs native mlx-lm on Qwen2 → `Δ = 0`.
- **MoE, bit-identical:** `StreamingSwitchGLU` vs native `SwitchGLU` on OLMoE →
  `max|Δ logits| = 0.0000`, both predict the same token.
- **Flagship, end-to-end:** full unquantized **GLM-5.2 (1.4 TB, 753B MoE)** on a
  24 GB M4 Pro: `The capital of France is` → **`Paris.`** Skeleton wired ~12 GB,
  712 s for prefill + 2 tokens, 465 GB streamed, **zero swap thrash**.

---

## Limitations (v0)

- Streaming is slow by design — tens of seconds to minutes per token on a ~1 GB/s
  drive. Full precision is the trade.
- Generic MoE supports **uniform-MoE** models only (mixed dense/MoE raises).
- GLM-5.2: context ≤ 2048 (the DSA sparse-attention indexer is bypassed — exact
  below `index_topk = 2048`); greedy decoding.
- Requires the model as safetensors with a `config.json`, fitting on disk.
