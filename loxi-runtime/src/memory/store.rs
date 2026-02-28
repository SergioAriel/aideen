// memory/store.rs
// ─────────────────────────────────────────────────────────────────────────────
// MemoryStore: HNSW-based vector database for long-term memory.
//
// Two indexes:
//   private: personal memories, never leave this node
//   shared:  community memories, synchronized with network opt-in
//
// Uses `instant-distance` crate (pure Rust, no C deps, compiles to WASM).
// HNSW parameters tuned for cognitive state embeddings (D_GLOBAL = 2048).
//
// Retrieval returns a list of MemoryEntry ranked by cosine similarity,
// which the backbone integrates as additional Δ_memory contributions.
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Embedding dimension must match D_GLOBAL in types.rs
pub const MEMORY_DIM: usize = 2048;

/// Max entries in shared index per node (prevents runaway growth)
pub const MAX_SHARED_ENTRIES: usize = 10_000;

/// Max entries in private index (per-node local storage)
pub const MAX_PRIVATE_ENTRIES: usize = 100_000;

/// Top-K entries to retrieve per query
pub const DEFAULT_TOP_K: usize = 8;

/// Minimum similarity threshold for retrieval (cosine similarity)
pub const MIN_SIMILARITY: f32 = 0.65;

// ─── Data Types ───────────────────────────────────────────────────────────────

/// Importancia de una entrada de memoria.
/// Determina comportamiento al actualizar el backbone y al replicar.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryImportance {
    /// Expira por LRU. No se re-embeds al actualizar backbone.
    Standard,
    /// Se re-embeds en background al actualizar backbone (v1→v2).
    /// Futuro P2P: replicada en 3 nodos.
    Important,
    /// Re-embeds prioritario + replicada con máxima redundancia.
    /// Futuro P2P: replicada en 5 nodos.
    Critical,
}

/// Scope de una entrada de memoria — alineado con types.rs MemoryScope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryScope {
    /// Solo RAM, dura mientras dure la sesión.
    Session,
    /// Disco local del nodo. Privada por defecto. Nunca sale sin permiso.
    Local,
    /// Replicada en red (hoy: BD Loxi encriptada; futuro: P2P).
    Distributed,
}

/// Contenido de una entrada de memoria.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryPayload {
    pub source_text: String,
    pub recall_text: String,
    pub domain: String,
    pub quality: f32,
    pub access_count: u32,
    pub created_at: u64,
    pub last_accessed: u64,
}

/// Una entrada de memoria a largo plazo del agente.
///
/// La memoria pertenece al AGENTE, no al nodo ni a la sesión.
/// Es portable — puede migrar entre nodos con el agente.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    /// Vector de embedding (clave para búsqueda HNSW).
    /// Dimensión = MEMORY_DIM = D_GLOBAL = 2048.
    pub embedding: Vec<f32>,
    /// Versión del backbone que generó este embedding.
    /// Al actualizar backbone, las entradas Important/Critical se re-embeds.
    /// "v1.0", "v1.1", "v2.0" — crítico para búsquedas cross-version.
    pub backbone_version: String,
    pub importance: MemoryImportance,
    pub payload: MemoryPayload,
    pub scope: MemoryScope,
}

impl MemoryEntry {
    pub fn new(
        embedding: Vec<f32>,
        backbone_version: String,
        importance: MemoryImportance,
        payload: MemoryPayload,
        scope: MemoryScope,
    ) -> Result<Self> {
        if embedding.len() != MEMORY_DIM {
            return Err(anyhow!(
                "embedding must be {} dims, got {}",
                MEMORY_DIM,
                embedding.len()
            ));
        }
        Ok(Self {
            id: Uuid::new_v4(),
            embedding,
            backbone_version,
            importance,
            payload,
            scope,
        })
    }

    /// Crea una entrada Standard/Local (el caso más común).
    pub fn standard(
        embedding: Vec<f32>,
        backbone_version: String,
        payload: MemoryPayload,
    ) -> Result<Self> {
        Self::new(
            embedding,
            backbone_version,
            MemoryImportance::Standard,
            payload,
            MemoryScope::Local,
        )
    }
}

/// Result of a memory retrieval query.
#[derive(Debug)]
pub struct RetrievalResult {
    pub entry: MemoryEntry,
    pub similarity: f32,
}

// ─── Simple Brute-Force Index (Development) ───────────────────────────────────
//
// Full production: use instant-distance HNSW.
// For initial development: brute-force cosine similarity.
// HNSW swap-in requires no API change — just replace LinearIndex with HnswIndex.
//
// LinearIndex is fine up to ~10K entries (query time <1ms).
// Above that, HNSW becomes necessary.

struct LinearIndex {
    entries: Vec<MemoryEntry>,
}

impl LinearIndex {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn insert(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    fn query(&self, query: &[f32], top_k: usize, min_similarity: f32) -> Vec<RetrievalResult> {
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (cosine_similarity(query, &e.embedding), i))
            .filter(|(sim, _)| *sim >= min_similarity)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(sim, idx)| RetrievalResult {
                entry: self.entries[idx].clone(),
                similarity: sim,
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn remove_oldest(&mut self, keep: usize) {
        if self.entries.len() > keep {
            // Sort by last_accessed ascending, keep the most recent
            self.entries.sort_by_key(|e| e.payload.last_accessed);
            self.entries.drain(0..self.entries.len() - keep);
        }
    }
}

// ─── Memory Store ─────────────────────────────────────────────────────────────

pub struct MemoryStore {
    node_id: String,
    private_index: Arc<RwLock<LinearIndex>>,
    shared_index: Arc<RwLock<LinearIndex>>,
    /// Path to persist private memory to disk
    private_path: PathBuf,
    /// Path to persist shared memory to disk
    shared_path: PathBuf,
}

impl MemoryStore {
    /// Create a new memory store for a given node.
    /// data_dir: where to persist entries (e.g. ~/.loxi/memory/{node_id}/)
    pub async fn new(node_id: &str, data_dir: &Path) -> Result<Self> {
        let private_path = data_dir.join(format!("{}_private.bin", node_id));
        let shared_path = data_dir.join(format!("{}_shared.bin", node_id));

        let mut store = Self {
            node_id: node_id.to_string(),
            private_index: Arc::new(RwLock::new(LinearIndex::new())),
            shared_index: Arc::new(RwLock::new(LinearIndex::new())),
            private_path,
            shared_path,
        };

        // Load persisted entries from disk
        store.load_from_disk().await;

        Ok(store)
    }

    // ── Insert ────────────────────────────────────────────────────────────────

    /// Store a new memory entry.
    /// Returns the UUID of the stored entry.
    pub async fn insert(&self, entry: MemoryEntry) -> Result<Uuid> {
        let id = entry.id;
        let scope = entry.scope.clone();

        match scope {
            MemoryScope::Session | MemoryScope::Local => {
                let mut idx = self.private_index.write().await;
                idx.insert(entry);
                // Enforce capacity limit
                idx.remove_oldest(MAX_PRIVATE_ENTRIES);
            }
            MemoryScope::Distributed => {
                let mut idx = self.shared_index.write().await;
                idx.insert(entry);
                idx.remove_oldest(MAX_SHARED_ENTRIES);
            }
        }

        Ok(id)
    }

    /// Convenience: create and store a memory from text + embedding.
    pub async fn remember(
        &self,
        embedding: Vec<f32>,
        source_text: String,
        recall_text: String,
        domain: String,
        quality: f32,
        scope: MemoryScope,
    ) -> Result<Uuid> {
        let now = unix_now();
        let entry = MemoryEntry::new(
            embedding,
            "v1.0".to_string(), // TODO: pull from backbone
            MemoryImportance::Standard,
            MemoryPayload {
                source_text,
                recall_text,
                domain,
                quality,
                access_count: 0,
                created_at: now,
                last_accessed: now,
            },
            scope,
        )?;
        self.insert(entry).await
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Retrieve the most relevant memories for a given query embedding.
    ///
    /// Searches both private and shared indexes.
    /// Results are ranked by cosine similarity, deduped, and trimmed to top_k.
    pub async fn retrieve(
        &self,
        query: &[f32],
        top_k: usize,
        min_sim: f32,
    ) -> Vec<RetrievalResult> {
        let top_k = top_k.min(DEFAULT_TOP_K * 2); // cap for safety

        // Search private and shared in parallel
        let private_idx = self.private_index.read().await;
        let shared_idx = self.shared_index.read().await;

        let mut results = Vec::new();
        results.extend(private_idx.query(query, top_k, min_sim));
        results.extend(shared_idx.query(query, top_k, min_sim));

        // Sort combined results by similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(top_k);

        results
    }

    /// Retrieve with default parameters.
    pub async fn retrieve_default(&self, query: &[f32]) -> Vec<RetrievalResult> {
        self.retrieve(query, DEFAULT_TOP_K, MIN_SIMILARITY).await
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    pub async fn private_count(&self) -> usize {
        self.private_index.read().await.len()
    }

    pub async fn shared_count(&self) -> usize {
        self.shared_index.read().await.len()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Persist all entries to disk (call periodically).
    pub async fn flush_to_disk(&self) -> Result<()> {
        let private_entries = {
            let idx = self.private_index.read().await;
            idx.entries.clone()
        };
        let shared_entries = {
            let idx = self.shared_index.read().await;
            idx.entries.clone()
        };

        // Serialize + lz4 compress
        let private_bytes = bincode_compress(&private_entries)?;
        let shared_bytes = bincode_compress(&shared_entries)?;

        tokio::fs::create_dir_all(self.private_path.parent().unwrap()).await?;
        tokio::fs::write(&self.private_path, &private_bytes).await?;
        tokio::fs::write(&self.shared_path, &shared_bytes).await?;

        tracing::debug!(
            "Memory flushed: {} private, {} shared entries",
            private_entries.len(),
            shared_entries.len()
        );

        Ok(())
    }

    async fn load_from_disk(&mut self) {
        if let Ok(bytes) = tokio::fs::read(&self.private_path).await {
            if let Ok(entries) = bincode_decompress::<Vec<MemoryEntry>>(&bytes) {
                let mut idx = self.private_index.write().await;
                for e in entries {
                    idx.insert(e);
                }
                tracing::info!("Loaded {} private memories from disk", idx.len());
            }
        }

        if let Ok(bytes) = tokio::fs::read(&self.shared_path).await {
            if let Ok(entries) = bincode_decompress::<Vec<MemoryEntry>>(&bytes) {
                let mut idx = self.shared_index.write().await;
                for e in entries {
                    idx.insert(e);
                }
                tracing::info!("Loaded {} shared memories from disk", idx.len());
            }
        }
    }

    // ── Shared sync helpers ───────────────────────────────────────────────────

    /// Export all shared entries for synchronization with the network.
    /// Called by the sync layer when Architect requests a snapshot.
    pub async fn export_shared(&self) -> Vec<MemoryEntry> {
        self.shared_index.read().await.entries.clone()
    }

    /// Merge incoming shared entries from a peer.
    /// Deduplication: entries with the same UUID are ignored.
    pub async fn merge_shared(&self, incoming: Vec<MemoryEntry>) -> usize {
        let mut idx = self.shared_index.write().await;
        let existing_ids: std::collections::HashSet<Uuid> =
            idx.entries.iter().map(|e| e.id).collect();

        let mut added = 0;
        for entry in incoming {
            if entry.scope == MemoryScope::Distributed && !existing_ids.contains(&entry.id) {
                idx.insert(entry);
                added += 1;
            }
        }
        idx.remove_oldest(MAX_SHARED_ENTRIES);
        added
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn bincode_compress<T: Serialize>(data: &T) -> Result<Vec<u8>> {
    let raw = bincode::serialize(data).map_err(|e| anyhow!("serialize error: {}", e))?;
    Ok(lz4_flex::compress_prepend_size(&raw))
}

fn bincode_decompress<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T> {
    let raw = lz4_flex::decompress_size_prepended(data)
        .map_err(|e| anyhow!("lz4 decompress error: {}", e))?;
    bincode::deserialize(&raw).map_err(|e| anyhow!("deserialize error: {}", e))
}
