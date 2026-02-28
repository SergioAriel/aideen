// memory/mod.rs
// ─────────────────────────────────────────────────────────────────────────────
// Long-term memory layer for Loxi — Hybrid architecture (Option C).
//
// Every node has:
//   - Private memory:  personal HNSW vector database, never shared
//   - Shared memory:   opt-in entries that can be queried by other nodes
//
// Memory entries are retrieved as CognitiveDelta contributions.
// A "memory node" is just a regular node that specializes in memory retrieval
// — it contributes Δ_memory to the integrated state just like any expert.
//
// Architecture:
//   MemoryStore
//     ├── private: HnswIndex     (local file, never exposed)
//     └── shared:  HnswIndex     (synchronized via Architect)
//
// Synchronization:
//   - Private: only ever written/read locally
//   - Shared: entries are broadcast to Architect, who gossips to peers
//             entries are compressed with lz4 before transmission
//
// Storage format:
//   - Each entry: { id, embedding [D_GLOBAL], payload: MemoryPayload }
//   - Stored as bincode + lz4 on disk
//   - HNSW index rebuilt on startup from persisted entries
//
// Design constraint: memory retrieval adds LATENCY to the query path.
// So retrieval runs in parallel with expert dispatch (async, non-blocking).
// ─────────────────────────────────────────────────────────────────────────────

pub mod store;
pub mod sync;

pub use store::{MemoryStore, MemoryEntry, MemoryPayload, MemoryScope, RetrievalResult};
pub use sync::{MemorySyncMessage, MemorySyncClient};
