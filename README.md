# RadixMesh: Distributed Radix Tree Cache for Accelerating LLM Serving

Use `SGLang`'s radix_cache and `mooncake` transfer engine to implement

## Run

```bash


```

## Why need RadixMesh

- Using distributed radix trees to share across `prefill` nodes to reuse kv cache, which accelerate LLM generation much
- Using distributed radix trees to share across `router` nodes to make cache-aware-routing

## How to be Distributed & Consistent

**Eventual Consistency**

- same base state
- ordered oplog applied
- **lock free** policy to resolve possible multi-write conflict

**Notice**

- The tree structures among the different nodes may not be identical, but the query (lookup) results should be
  consistent.

## Future Roadmap

- node failure detection and reorg topo
- register callback for receive oplog rather than polling
- exchange addr of remote natively

## Detail
### Limitation

- All nodes' config should be same except `local_cache_addr`

