# Hot-expert cache — the "sliding window" memory for routed experts.
#
# MoE expert access is CYCLIC BY LAYER: a token touches layer 3's experts, then
# 4's, ... then 77's; the next token starts again at layer 3. A single global LRU
# smaller than one token's whole expert working set (75 layers x 8) gets fully
# overwritten before you cycle back to a layer, so cross-token reuse is lost.
#
# Therefore the cache is PARTITIONED PER LAYER: each MoE layer gets its own LRU,
# so layer L's hottest experts survive across tokens regardless of what the other
# layers do. This is what actually captures expert locality. Cached arrays stay
# referenced here, so mx.clear_cache() won't free them.

from collections import OrderedDict


class _LayerLRU:
    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.store: "OrderedDict[int, dict]" = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, e: int):
        v = self.store.get(e)
        if v is not None:
            self.store.move_to_end(e)
            self.hits += 1
            return v
        self.misses += 1
        return None

    def put(self, e: int, tensors: dict):
        if e in self.store or self.budget <= 0:
            return
        nb = sum(a.nbytes for a in tensors.values())
        if nb > self.budget:
            return
        self.store[e] = tensors
        self.bytes += nb
        while self.bytes > self.budget and self.store:
            _, ev = self.store.popitem(last=False)
            self.bytes -= sum(a.nbytes for a in ev.values())


class LayeredExpertCache:
    """Per-layer LRU partition of a shared byte budget."""

    def __init__(self, total_budget_bytes: int, moe_layers):
        moe_layers = list(moe_layers)
        per = total_budget_bytes // max(len(moe_layers), 1)
        self.per_layer_budget = per
        self.caches = {l: _LayerLRU(per) for l in moe_layers}

    def get(self, l: int, e: int):
        return self.caches[l].get(e)

    def put(self, l: int, e: int, tensors: dict):
        self.caches[l].put(e, tensors)

    @property
    def hits(self):
        return sum(c.hits for c in self.caches.values())

    @property
    def misses(self):
        return sum(c.misses for c in self.caches.values())

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0

    def experts_per_layer(self) -> float:
        if not self.caches:
            return 0.0
        return sum(len(c.store) for c in self.caches.values()) / len(self.caches)
