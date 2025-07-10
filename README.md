# RadixMesh: Distributed Radix Tree Cache for Accelerating LLM Serving

Based on `SGLang`'s radix_cache to implement

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

## Why need RadixMesh

- Using distributed radix trees to share across `prefill/decode` nodes to reuse kv cache, which accelerate LLM generation much
- Using distributed radix trees to share across `router` nodes to make cache-aware-routing

## How to be Distributed & Consistent

**Eventual Consistency**

- same base state
- ordered oplog applied
- **lock free** policy to resolve possible multi-write conflict

**Notice**

- The tree structures among the different nodes may not be identical, but the query (lookup) results will be
  consistent.

## Future Roadmap

- [ ] node failure detection and topo update
    - [ ] dynamic new nodes
- [ ] use `rdma` lib as fast communicator? like `mooncake-transfer-engine`?
    - [ ] register callback for receive oplog rather than polling
    - [ ] exchange remote addr of remote natively
- [ ] oplog msg priority

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
- Double loop detection

### cache-aware routing

- prefix match first
    - give prefill node & decode node
- consistent hash if no nodes has cache

### Limitation

- All nodes' config should be same except `local_cache_addr`

