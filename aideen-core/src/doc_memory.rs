use serde::{Deserialize, Serialize};

pub type DocId = u64;
pub type ChunkId = u32;

/// Canonical document metadata.
///
/// `locator` is a stable identifier of the origin:
/// - native: absolute/relative path
/// - web:    URL
/// - ui:     "chat:<id>" or "note:<id>"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub title: String,
    pub locator: String,
    pub mime: String, // "text/plain", "application/pdf", etc.
    pub len_bytes: u64,
    pub added_unix: u64,
}

/// Result of a lexical search.
///
/// `byte_start/byte_end` are offsets in the raw document (exclusive on end,
/// Rust-style `[byte_start..byte_end]`).
/// `preview` is best-effort text for UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocHit {
    pub doc_id: DocId,
    pub chunk_id: ChunkId,
    pub score: f32,
    pub byte_start: u64,
    pub byte_end: u64,
    pub preview: String,
}

/// Minimal contract for document memory (precise text search).
///
/// Design:
/// - Exact lexical search: token → posting list, ranking by TF.
/// - `locate()`: byte-exact offsets of a needle in the raw document.
/// - `get_chunk()`: actual bytes of the chunk (for building prompts).
///
/// Backends: `FsDocMemory` (native), `OpfsDocMemory` (WASM).
pub trait DocMemory {
    /// Inserts a complete document (raw bytes). Returns the assigned `DocId`.
    /// Overwrites `meta.len_bytes` with `bytes.len()`.
    fn add_document(&mut self, meta: DocMeta, bytes: Vec<u8>) -> Result<DocId, String>;

    /// Returns metadata if the `doc_id` exists.
    fn get_meta(&self, doc_id: DocId) -> Option<DocMeta>;

    /// Retrieves bytes of a specific chunk (real context for prompts).
    fn get_chunk(&self, doc_id: DocId, chunk_id: ChunkId) -> Option<Vec<u8>>;

    /// Lexical search. Returns top-k hits sorted by descending score.
    fn search(&self, query: &str, k: usize) -> Vec<DocHit>;

    /// Returns offsets `(byte_start, byte_end)` of exact occurrences of
    /// `needle` in the raw document. `byte_end` is exclusive. No stemming.
    fn locate(&self, doc_id: DocId, needle: &[u8], limit: usize) -> Vec<(u64, u64)>;

    /// Lists all stored doc_ids.
    fn list_docs(&self) -> Vec<DocId>;
}

/// No-op backend (no document memory).
/// Allows starting the node without a configured DocMemory.
pub struct NullDocMemory;

impl DocMemory for NullDocMemory {
    fn add_document(&mut self, _meta: DocMeta, _bytes: Vec<u8>) -> Result<DocId, String> {
        Ok(0)
    }
    fn get_meta(&self, _doc_id: DocId) -> Option<DocMeta> {
        None
    }
    fn get_chunk(&self, _doc_id: DocId, _chunk_id: ChunkId) -> Option<Vec<u8>> {
        None
    }
    fn search(&self, _query: &str, _k: usize) -> Vec<DocHit> {
        vec![]
    }
    fn locate(&self, _doc_id: DocId, _needle: &[u8], _limit: usize) -> Vec<(u64, u64)> {
        vec![]
    }
    fn list_docs(&self) -> Vec<DocId> {
        vec![]
    }
}
