# RadixMesh: Distributed Radix Tree Cache for Accelerating LLM Serving

This project is implemented based on the `radix_cache` from the `SGLang` project.

<img alt="Image" src="https://github.com/user-attachments/assets/76361a33-ddf3-4184-9ac3-a6dab1401940" />

## Why need RadixMesh

- `RadixMesh` can reuse kv cache in any routing policy.
- `RadixMesh` can gc dup kv cache.
- `RadixMesh` can be used by prefill or decode node to redirect req to other node if overloaded.
- Updating the routing rule based on real-time event makes `RadixMesh` faster.
- `RadixMesh` is capable of finding best match for both prefill and decode nodes.

## Run

### Config

1. (optional) venv

```bash
python3 -m venv venv
source venv/bin/activate
```

2. requirements

```bash
cd python
pip3 install .
```

### Correctness Test

```bash
#cd python
python3 -m src.test.correctness
```

### Benchmark

```bash
#cd python
python3 -m src.test.benchmark
```

## Future Roadmap

- [ ] node failure detection and topo update
    - [ ] dynamically add/remove nodes
- [ ] use `rdma` lib as fast communicator? like `mooncake-transfer-engine`?
    - [ ] register callback for receive oplog rather than polling
    - [ ] exchange remote addr of remote natively
- [ ] oplog msg priority
- [ ] pb serializer
- [ ] `send` change to async operation
- [ ] better topo if nodes over some number (like 50?)
- [ ] benchmark perf

## How to be Distributed & Consistent

**Eventual Consistency**

- same base state
- ordered oplog applied
- oplog is idempotent
- **master free** policy to resolve possible multi-write conflict

**Notice**

- The tree structures among the different nodes may not be identical, but the query (lookup) results will be
  consistent.

## Tech Detail

### tcp connection with simple protocol

```
 4-byte
[length][data]
```

### keep alive

- Send heartbeat requests at regular intervals to keep the connection active and detect topology anomalies.
- Heartbeat with ttl to avoid infinite loop
- Hierarchy heartbeat
    - next node short heartbeat
    - ring check with long gap

### node topo check

- Two-rounds verification for ring-topo

### cache-aware routing

- warmup stage use random hash routing strategy to avoid all request go only one node
- prefix match first
    - give prefill node & decode node
- consistent hash if no nodes has cache

### gc dup kv cache

- [x] local ref
    - When to gc: periodical
    - How to gc
        - each tree send try gc req, collect agree
        - if all agree, gc
- [ ] global ref
    - When to gc: periodical & ref reach 0
    - How to gc
        - insert oplog with initial ref count
        - update ref count when used/freed (idempotent impl is complex)
        - when ref reach 0, send try gc req, collect agree
        - each tree send try gc req, collect agree
        - if all agree, gc

### same tree but diff value

- router value & prefill value & normal value have different value class

## Limitation

- All nodes' config should be same except `local_cache_addr`

