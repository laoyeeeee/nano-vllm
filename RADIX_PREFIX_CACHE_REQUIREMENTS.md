# Radix Prefix Cache Requirements

## 1. Background

Nano-vLLM currently implements prefix caching in `BlockManager` with a rolling hash table:

```python
hash_to_block_id: dict[int, int]
```

Each complete KV cache block stores:

- `hash`: rolling hash of the current block and previous prefix hash.
- `token_ids`: tokens contained in the block.
- `ref_count`: number of active sequences using this block.

This design is simple and effective, but prefix reuse is effectively block-granular. With the default block size of 256, two prompts sharing 300 tokens can safely reuse the first 256-token block, while the remaining shared 44 tokens still need to be recomputed.

Radix prefix caching is useful for improving prefix lookup semantics and preparing the project for SGLang-style prefix cache behavior.

## 2. Goals

- Add a radix-tree prefix index for token-prefix lookup.
- Support longest-prefix matching before allocating new KV cache blocks.
- Keep the first implementation block-granular for KV reuse.
- Preserve the existing paged KV cache layout and `block_table` mechanism.
- Preserve current sequence preemption, block ref counting, and deallocation semantics.
- Provide clear metrics for cache hit length and reused block count.

## 3. Non-Goals

- No partial-block KV reuse in the first version.
- No eviction policy in the first version unless required by current free-block behavior.
- No cross-process or distributed radix cache.
- No persistent cache across engine restarts.
- No changes to streaming output behavior.
- No multimodal prefix hashing in the first version.

## 4. Key Design Decision

The first version should use a radix tree only as a **prefix index**, while keeping actual KV reuse aligned to full KV cache blocks.

Example:

```text
prompt A and prompt B share 300 tokens
block_size = 256
```

Radix longest-prefix match may discover 300 shared tokens, but the reusable KV cache length should be rounded down to 256 tokens. The remaining 44 tokens are recomputed.

This keeps the implementation compatible with the current block table design:

```text
seq.block_table = [block_id_0, block_id_1, ...]
seq.num_cached_tokens = reused_full_blocks * block_size
```

## 5. Proposed Data Model

Add a radix cache component owned by `BlockManager`.

Suggested classes:

```python
class RadixNode:
    token_ids: list[int]
    children: dict[int, RadixNode]
    block_ids: list[int]
    ref_count: int
```

```python
class RadixPrefixCache:
    root: RadixNode

    def match(self, token_ids: list[int]) -> tuple[int, list[int]]:
        ...

    def insert(self, token_ids: list[int], block_ids: list[int]):
        ...

    def remove(self, block_ids: list[int]):
        ...
```

Return values for `match(...)`:

- `matched_tokens`: length of the longest token prefix found.
- `block_ids`: reusable block ids covering only complete blocks.

The radix tree may store token spans instead of one token per node to reduce memory overhead.

## 6. Integration Plan

### 6.1 Allocation

Modify `BlockManager.allocate(seq)`:

1. Query radix prefix cache with `seq.prompt_token_ids` or `seq.token_ids`.
2. Reuse only complete blocks from the matched prefix.
3. Increment `ref_count` for reused blocks.
4. Set `seq.num_cached_tokens` to the reused full-block token count.
5. Allocate new blocks for the remaining sequence blocks.
6. Insert newly completed blocks into the radix cache.

The existing `seq.block_table` remains the source of truth for attention.

### 6.2 Append During Decode

Keep current append behavior:

- Allocate a new block when the current sequence crosses a block boundary.
- Once a block becomes complete, insert or update it in the radix cache.
- Do not expose partial blocks for reuse.

### 6.3 Deallocation

`BlockManager.deallocate(seq)` should continue decrementing block ref counts.

Radix cache entries should remain valid only for blocks whose KV cache content is still available. If a block is returned to the free list and later reused for different tokens, stale radix entries must not point to it.

Recommended first-version rule:

- Remove or invalidate radix entries when their backing block `ref_count` reaches 0 and the block returns to `free_block_ids`.

This avoids stale prefix hits at the cost of lower cache persistence.

## 7. Correctness Requirements

- A radix hit must verify token equality before reusing blocks.
- Reused length must be a multiple of `block_size`.
- `seq.num_cached_tokens` must match the number of reused tokens.
- `seq.block_table` order must exactly match the reused prefix block order.
- A block id must never be reused from radix cache after the block has been freed and repurposed.
- Prefix cache behavior must be identical for prefill, chunked prefill, and preemption re-prefill from the model's perspective.

## 8. Metrics and Debugging

Add lightweight counters on `BlockManager`:

- `radix_cache_queries`
- `radix_cache_hits`
- `radix_cache_hit_tokens`
- `radix_cache_hit_blocks`
- `radix_cache_inserts`
- `radix_cache_invalidations`

These counters are useful for comparing:

- Existing hash-based prefix cache.
- Radix block-granular prefix cache.
- Future partial-block radix cache.

## 9. Testing Requirements

### 9.1 Basic Prefix Reuse

- Two prompts with identical first full block reuse that block.
- `seq.num_cached_tokens == block_size`.
- The second prompt output matches the baseline without prefix cache.

### 9.2 Longest Prefix Match

- Prompts sharing multiple full blocks reuse all shared full blocks.
- Prompts diverging inside a block only reuse previous complete blocks.
- Prompts sharing fewer than `block_size` tokens do not reuse KV blocks in v1.

### 9.3 Safety

- Freed blocks are not returned by stale radix entries.
- Preemption followed by re-prefill does not corrupt `block_table`.
- Chunked prefill with radix hits produces the same output as no-cache prefill.

### 9.4 Compatibility

- Existing `example.py` still runs.
- Existing `bench.py` still runs.
- Tensor parallel smoke test passes if multi-GPU is available.
- Streaming output, once implemented, remains unaffected because radix cache only changes prefill reuse.

## 10. Acceptance Criteria

- Radix prefix cache can be enabled without changing public `LLM` APIs.
- Full-block shared prefixes are reused through radix longest-prefix lookup.
- Partial-block matches are detected but rounded down to full-block reuse.
- No stale radix entry can reference a freed and repurposed block.
- Outputs match the existing hash-cache implementation for equivalent prompts.
- Cache metrics make reuse behavior observable.

## 11. Future Work

- Partial-block KV reuse.
- LRU or priority-based prefix cache eviction.
- Prefix cache visualization.
- Request-aware cache pinning for shared system prompts.
- Multimodal-aware cache keys that include image/video content hashes.
- Distributed radix cache for disaggregated prefill/decode.
