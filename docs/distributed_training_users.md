# AIDEEN Distributed Training on User Machines

## Scope

This document defines the recommended strategy for training AIDEEN across heterogeneous user-owned machines, without relying on a tightly synchronized dedicated cluster.

The target is:
- one global model,
- many unreliable/heterogeneous nodes,
- intermittent connectivity,
- no assumption of low-latency all-reduce,
- controlled convergence and stability.

This document is intentionally separate from local GPU performance notes in
[docs/perf_bottlenecks.md](/Users/sergiosolis/Programacion/AIDEEN/docs/perf_bottlenecks.md).

---

## Recommendation

The recommended architecture is:

- **federated asynchronous rounds**
- **central coordinator for aggregation and validation**
- **short local training windows**
- **signed checkpoints / signed accepted updates**
- **global model updated by rounds, not by step-level synchronous all-reduce**

This is the best fit for user machines because:
- nodes are heterogeneous,
- nodes disconnect,
- latency is high and unstable,
- strict step-by-step synchronization is impractical,
- AIDEEN's DEQ solver is more sensitive to drift than a trivial stateless training loop.

---

## What Not To Do

### 1. Not recommended: synchronous data parallel per step

Do not start with:
- all-reduce every step,
- strict synchronization across user machines,
- waiting for the slowest node.

Why:
- domestic internet latency dominates,
- stragglers stall everyone,
- user machines are not homogeneous,
- operational complexity is too high for a first viable system.

### 2. Not recommended: train independently and average once at the end

Do not rely on:
- fully independent local training,
- one final weight average after long divergence.

Why:
- replicas drift too far,
- convergence degrades,
- DEQ-sensitive dynamics may diverge silently,
- final average may be mathematically poor even if local runs looked healthy.

---

## Recommended Training Model

### Global round-based protocol

Each round:

1. Coordinator publishes `checkpoint_t`
2. Client downloads `checkpoint_t`
3. Client trains locally for a short bounded budget
4. Client uploads:
   - weight delta or gradient delta
   - tokens processed
   - validation metrics
   - DEQ stability metrics
   - run metadata
5. Coordinator filters invalid updates
6. Coordinator aggregates accepted updates
7. Coordinator publishes `checkpoint_t+1`

This yields one global model while tolerating heterogeneous participants.

---

## Local Node Budget

Recommended local work per round:

- **small bounded number of steps**
- not hours of drift
- enough tokens to amortize startup overhead

Practical first target:
- `10` to `100` microsteps per node per round
- fixed config
- fixed checkpoint base

Reason:
- enough signal to contribute meaningful gradient information
- small enough to limit divergence between rounds
- small enough to discard bad updates cheaply

---

## What Each Node Should Report

Each node should upload at minimum:

- `base_checkpoint_hash`
- `delta_weights_hash`
- `tokens_processed`
- `loss`
- `val_loss` if available
- `contr`
- `max_delta`
- `avg_iters`
- `invalid_fixed_point_count`
- `nan_detected`
- `device metadata`
- `training config hash`

For AIDEEN specifically, the DEQ-related metrics are mandatory because convergence quality is part of model correctness, not just observability.

---

## Acceptance Rules

Coordinator should reject updates that show any of:

- wrong `base_checkpoint_hash`
- NaNs or infinities
- invalid config hash
- `DEQ-INVALID`
- anomalous `max_delta`
- anomalous `contr`
- suspicious delta norm
- malformed or unsigned payload

Coordinator should clip or downweight updates with:

- too few tokens,
- weak stability,
- very low hardware reliability,
- extreme outlier gradients.

---

## Aggregation Strategy

Start with:

- weighted average of deltas
- weight by accepted token count
- optional delta clipping before aggregation

Recommended first aggregation:

1. verify update validity
2. clip per-update norm
3. weight by tokens processed
4. average deltas
5. apply to global checkpoint

Do not start with exotic federated optimizers first. Get the system correct and measurable before introducing extra aggregation complexity.

---

## Security / Trust Model

Because the nodes are user-owned:

- checkpoints should be signed,
- accepted updates should be signed by the coordinator,
- clients should not define global truth,
- the coordinator decides which updates become part of the official model lineage.

This is consistent with the existing protocol direction in
[docs/protocol_v1.md](/Users/sergiosolis/Programacion/AIDEEN/docs/protocol_v1.md).

---

## Why This Fits AIDEEN

AIDEEN is not a simple feed-forward trainer. It carries:

- DEQ forward convergence constraints
- staged Picard adjoint
- history path and recurrence-sensitive dynamics

That means:
- local updates must be short and validated,
- aggregation must be conservative,
- metrics about convergence are part of acceptance,
- asynchronous federated rounds are safer than loose uncontrolled local drift.

---

## Minimal Viable Architecture

### Phase 1 — Coordinator MVP

Components:

- `coordinator`
- `node client`
- signed checkpoint distribution
- upload of local deltas + metrics
- aggregator

Capabilities:

- one official model lineage
- bounded local rounds
- accepted/rejected updates

### Phase 2 — Reliability Layer

Add:

- retries
- resumable uploads
- node reputation
- stale checkpoint rejection
- weighted trust for updates

### Phase 3 — Production Federation

Add:

- scheduling
- optional dataset/task assignment
- quota control
- rate limits
- abuse resistance
- better aggregation policies

---

## Implementation Estimate

These estimates assume one engineer who already understands the codebase.

### MVP for controlled experiments

Goal:
- coordinator
- local round execution
- upload deltas
- aggregate
- publish new checkpoint

Estimate:
- **2 to 4 weeks**

This is realistic if:
- single coordinator,
- no browser support first,
- trusted small node set,
- no advanced security hardening yet.

### Robust internal alpha

Goal:
- unstable user machines tolerated
- checkpoint lineage validation
- update filtering
- metrics storage
- rejections and retries

Estimate:
- **4 to 8 weeks**

This is the first version I would consider operationally meaningful for many volunteer machines.

### Public heterogeneous network

Goal:
- many nodes,
- untrusted participants,
- failure handling,
- abuse resistance,
- secure rollout discipline,
- robust coordinator behavior,
- monitoring and rollback

Estimate:
- **2 to 4 months**

This is the realistic range if you want something that can survive internet conditions and bad participants without corrupting the model.

---

## Does “Many Nodes” Make It Faster?

Only if aggregation is controlled.

With many nodes:
- throughput of total processed tokens can scale,
- coordinator complexity also rises,
- invalid update filtering becomes more important,
- straggler tolerance matters,
- communication and checkpoint management become the real bottleneck.

So:
- more nodes do not automatically reduce implementation time,
- they mainly increase systems engineering requirements.

The right expectation is:
- more nodes can increase training capacity,
- but they also require stronger coordination logic.

---

## Practical Recommendation

Start with:

- one coordinator,
- native clients only,
- small trusted node set,
- short local rounds,
- conservative aggregation,
- strict metric-based rejection.

Do not start with:

- browser federation,
- P2P-only coordination,
- end-of-training weight merge,
- per-step synchronous distributed training over the internet.

---

## Suggested Next Engineering Steps

1. Define the update payload:
   - checkpoint hash
   - delta format
   - metrics
   - signatures

2. Define the coordinator acceptance policy:
   - rejection criteria
   - clipping
   - weighting

3. Define local round budget:
   - steps
   - tokens
   - fixed config

4. Implement coordinator MVP

5. Run closed federation with a few trusted nodes before opening the network

---

## Repository-Level Design

This section maps the federated training design to concrete implementation work in the AIDEEN workspace.

### Proposed crates / modules

Recommended minimum split:

- `aideen-coordinator`
  - round management
  - checkpoint publication
  - update intake
  - validation
  - aggregation

- `aideen-node` or client-side training module
  - download checkpoint
  - run local round
  - package metrics + deltas
  - upload signed update

- shared protocol module
  - message types
  - serialization
  - hashes
  - signatures

If protocol pieces already live near
[docs/protocol_v1.md](/Users/sergiosolis/Programacion/AIDEEN/docs/protocol_v1.md),
reuse that direction instead of creating a second incompatible protocol surface.

---

## Minimum Protocol Messages

At minimum, the coordinator flow needs:

- `CheckpointOffer`
  - checkpoint hash
  - checkpoint version
  - config hash
  - round id

- `CheckpointAck`
  - node accepted round base

- `LocalUpdate`
  - base checkpoint hash
  - delta payload
  - metrics
  - tokens processed
  - node metadata
  - optional signature

- `UpdateAccepted`
  - accepted / rejected
  - reason
  - next round info

- `CheckpointPublished`
  - new global checkpoint hash
  - version
  - aggregation metadata

This is enough for an MVP coordinator.

---

## Delta Representation

For a first implementation, use:

- dense delta over trainable weights
- same layout/order as the existing GPU/CPU checkpoint export

Do not start with sparse or compressed updates unless bandwidth becomes the first blocker.

### Practical first format

For each round:
- `base_checkpoint_hash`
- `n_tokens`
- `delta_blob`
- `metrics_blob`

Where:
- `delta_blob` is the flat parameter delta in the same canonical order as checkpoint serialization
- `metrics_blob` includes DEQ stability fields

### Why canonical layout matters

If each node exports deltas in a slightly different order, aggregation becomes fragile and silent corruption becomes likely.

The delta format must be tied to:
- checkpoint version
- architecture config hash
- parameter ordering hash

---

## AIDEEN-Specific Metrics Required In Update Payload

Besides generic ML fields, each local round should report:

- `avg_loss`
- `val_loss` if computed
- `avg_contr`
- `max_delta`
- `avg_iters`
- `invalid_fixed_point_count`
- `nan_count`
- `deq_config_hash`
- `training_config_hash`

These are not optional diagnostics. They are part of the acceptance contract for DEQ-based training.

---

## Local Round Execution Contract

Each node must run with:

- fixed checkpoint base
- fixed config bundle
- bounded local step budget
- deterministic report format

Coordinator should reject updates produced with:

- mismatched config
- mismatched checkpoint base
- unsupported architecture version

This avoids averaging updates that do not belong to the same model state.

---

## Suggested MVP Implementation Order

### Step 1 — Canonical checkpoint hash + config hash

Before federation, ensure:
- stable checkpoint export
- stable parameter ordering
- stable hashing

Without this, distributed aggregation is unsafe.

### Step 2 — Local round runner

Implement a node-side function like:

- load checkpoint
- train `N` local steps
- export delta + metrics

This can exist first as a local CLI command before any network transport.

### Step 3 — Coordinator aggregator

Coordinator:
- receives local updates
- validates hashes/metrics
- clips and weights
- emits new checkpoint

Again, this can first run offline on local files before turning into a network service.

### Step 4 — Network transport

Only after local file-based federation works:
- add transport
- upload/download flow
- retry and fault handling

This order reduces debugging complexity significantly.

### Step 5 — Hardening

Then add:
- signatures
- replay protection
- stale round rejection
- node reputation or trust weighting

---

## Strong Recommendation On Implementation Strategy

Do not start by writing the whole networked version.

Build in this order:

1. single-machine simulated federation
2. multi-process local federation
3. coordinator over network
4. heterogeneous user nodes

That sequence proves the training logic before adding unreliable transport.

---

## What To Prototype First In Code

If implementation starts soon, the first code prototype should be:

- a local `round_runner`
- a local `aggregate_updates`
- a canonical `export_delta_from_checkpoint_pair`

These three pieces answer the real hard question first:

- can AIDEEN be safely aggregated round by round without damaging convergence?

That is more important than transport at the beginning.
