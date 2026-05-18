# AIDEEN Network Protocol v1

> **Status**: Frozen — compatible with `aideen-core >= 0.3`
> **Wire version**: `Hello.protocol = 1`

---

## 1. Framing

All messages use the same binary framing, both in QUIC and in WebTransport:

```
┌──────────────┬────────────────────────────────────┐
│  length (4B) │  payload (bincode)                  │
│  u32 LE      │  NetMsg (serde / bincode)            │
└──────────────┴────────────────────────────────────┘
```

- **4 bytes**: payload length in little-endian (`u32`).
- **Payload**: `bincode::serialize(&NetMsg)` — no versioning of its own, the enum discriminant acts as the type tag.
- No padding. No final delimiter.

### Transport

| Transport | Stream     | Direction        |
|------------|------------|------------------|
| QUIC       | uni stream | C→S: client opens; S→C: server opens |
| WebTransport (future) | unidirectional stream | same |
| InProcess (tests)  | `sync_channel(64)` | — |

Two independent uni streams eliminate the initialization deadlock: each side opens its own outgoing stream and accepts the one from the other side, regardless of who writes first.

---

## 2. Enum `NetMsg`

### `Hello`
```
Hello {
    node_id:        [u8; 32],   // Ephemeral public key of the node
    protocol:       u32,        // = 1 for this version
    bundle_version: u64,        // Installed model bundle
    bundle_hash:    [u8; 32],   // SHA-256 of the bundle (topological alignment)
}
```
First message from the client when opening a session. The server responds with `Delegation`.

### `Delegation(KeyDelegation)`
```
KeyDelegation {
    epoch:               u64,
    critic_pk:           [u8; 32],
    valid_from_unix:     u64,
    valid_to_unix:       u64,
    signature_by_root:   Vec<u8>,   // ed25519 over signing_bytes()
}
```
The coordinator delegates the operational key of the current Critic. The signature is verified against the `root_pk` embedded in the node. It is only accepted if `epoch > current_epoch` (anti-rollback).

### `Ack`
```
Ack {
    kind:    AckKind,   // Delegation | Update | Discovery
    version: u64,       // epoch (Delegation) or version (Update/Discovery)
    ok:      bool,
}
```
Typed confirmation. The `kind` field lets the coordinator verify that the client confirmed the correct message without ambiguity.

### `Update(SignedUpdate)`
```
SignedUpdate {
    version:          u64,
    target_id:        String,      // expert that receives the delta
    bundle_version:   u64,
    bundle_hash:      [u8; 32],
    base_model_hash:  [u8; 32],    // snapshot the delta applies to
    prev_update_hash: [u8; 32],    // anti-fork chaining
    payload:          Vec<u8>,     // zstd( bincode( Vec<QuantizedDelta> ) )
    signature:        Vec<u8>,     // ed25519 over signing_bytes()
}
```
The client verifies signature + `base_model_hash` + anti-replay (`version > last_seen_version`) before applying.

### `Discovery`
```
Discovery {
    node_id:        [u8; 32],
    target_id:      String,
    q_total:        f32,
    iters:          u32,
    stop:           u8,          // 0 = Q_MIN_WRITE gate, 1 = Epsilon gate
    h_star_hash:    [u8; 32],    // SHA-256 of h* serialized in LE
    context_hash:   [u8; 32],    // SHA-256 of s_context
    bundle_version: u64,
}
```
Quality signal sent by the node when `Q >= Q_MIN_LEARN`. **It does not contain the full `h*`** — only hashes for correlation, dedup and anti-spam. The Architect decides whether to request more information from the node.

### `ExpertTask` / `ExpertResult`
```
ExpertTask {
    task_id:        [u8; 16],
    target_id:      String,
    s_r:            Vec<f32>,    // reasoning subspace (~8KB)
    bundle_version: u64,
}

ExpertResult {
    task_id:  [u8; 16],
    target_id: String,
    delta:    Vec<f32>,    // Δ = h_next − h (~8KB)
    q_total:  f32,
    iters:    u32,
    stop:     u8,
}
```
Peer-to-peer inference task routing between expert nodes.

### `Metrics`
```
Metrics { q_total: f32, iters: u32, stop: u8 }
```
Lightweight telemetry, does not generate Discovery. Used for monitoring without triggering the learning cycle.

### `Error`
```
Error { code: u32, msg: String }
```
Codes:
- `400` — unexpected message in the current session state.
- `403` — Discovery received without a prior Delegation (Zero-Trust).

### `Ping` / `Pong`
Heartbeat. The coordinator responds with `Pong` to any `Ping` in any state.

---

## 3. Coordinator state machine

```
        Hello
[Waiting] ──────────────► [HelloReceived]
                                │
                           Ack(Delegation)
                                │
                                ▼
                          [Delegated]
                           │        │
                    Discovery       Ack(Update)
                     → Ack       → session complete
```

Out-of-sequence transitions generate `Error { code: 400 }` and close the session.

---

## 4. Protocol invariants

1. **No `h*` on the wire**: `Discovery` only carries `h_star_hash` (SHA-256). The full vector never leaves the node via this route.

2. **Anti-replay in Delegation**: `epoch` must be strictly greater than `current_epoch`. The node rejects delegations with an equal or lower epoch.

3. **Anti-replay in Update**: `version` must be greater than the last version applied for that `target_id`. The `ReplayGuard` keeps the per-target map.

4. **Update chaining**: `prev_update_hash` creates a chain that the Critic can verify to detect forks or malicious reorderings.

5. **Zero-Trust**: `Discovery` before `Delegation` → `Error { code: 403 }`. The coordinator does not accept intelligence without prior cryptographic authorization.

6. **Double signature**: Delegation (signed by root_sk) + Update (signed by the delegated critic_sk). The node verifies both chains before applying changes.

---

## 5. Future compatibility with WebTransport

WebTransport over HTTP/3 uses the same underlying QUIC mechanism. The migration requires:

- Replacing `quinn::SendStream` / `RecvStream` with `web_transport::SendStream` / `RecvStream`.
- The framing (u32 LE + bincode) remains **identical** — there is no protocol change.
- The `Hello.protocol = 1` discriminant allows negotiating future versions without breaking compatibility.
- In WASM, WebTransport replaces native QUIC; the `NetMsg` enum and the framing are the same.

---

## 6. Version log

| Wire version | Main change |
|---|---|
| 1 | Initial: Hello + Delegation + Update + Ack(AckKind) + Discovery + Error |

---

## Appendix A — Distributed inference design note

`ExpertTask` / `ExpertResult` are the v1 MVP wire shape for expert deltas. The current architectural target for distributed inference is broader: experts should preferably return structured memory/context/evidence patches that are locally gated, integrated, and refined before the LMHead decodes.

See [docs/vision/distributed_inference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/distributed_inference.md) for the current design direction. This appendix does not change wire version `1`; richer `ExpertQuery` / `ExpertPatch` messages should be introduced only through an explicit future protocol version.
