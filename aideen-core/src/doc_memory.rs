use serde::{Deserialize, Serialize};

pub type DocId = u64;
pub type ChunkId = u32;

/// Metadata canónica de un documento.
///
/// `locator` es un identificador estable del origen:
/// - native: path absoluto/relativo
/// - web:    URL
/// - ui:     "chat:<id>" o "note:<id>"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub title: String,
    pub locator: String,
    pub mime: String, // "text/plain", "application/pdf", etc.
    pub len_bytes: u64,
    pub added_unix: u64,
}

/// Resultado de una búsqueda lexical.
///
/// `byte_start/byte_end` son offsets en el documento raw (exclusivo en end,
/// al estilo Rust `[byte_start..byte_end]`).
/// `preview` es texto best-effort para UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocHit {
    pub doc_id: DocId,
    pub chunk_id: ChunkId,
    pub score: f32,
    pub byte_start: u64,
    pub byte_end: u64,
    pub preview: String,
}

/// Contrato mínimo de memoria documental (búsqueda precisa por texto).
///
/// Diseño:
/// - Búsqueda lexical exacta: token → posting list, ranking por TF.
/// - `locate()`: offsets byte-exactos de una needle en el documento raw.
/// - `get_chunk()`: bytes reales del chunk (para construir prompts).
///
/// Backends: `FsDocMemory` (native), `OpfsDocMemory` (WASM).
pub trait DocMemory {
    /// Inserta documento completo (bytes raw). Devuelve `DocId` asignado.
    /// Sobrescribe `meta.len_bytes` con `bytes.len()`.
    fn add_document(&mut self, meta: DocMeta, bytes: Vec<u8>) -> Result<DocId, String>;

    /// Devuelve metadata si el `doc_id` existe.
    fn get_meta(&self, doc_id: DocId) -> Option<DocMeta>;

    /// Recupera bytes de un chunk específico (contexto real para prompts).
    fn get_chunk(&self, doc_id: DocId, chunk_id: ChunkId) -> Option<Vec<u8>>;

    /// Búsqueda lexical. Devuelve top-k hits ordenados por score descendente.
    fn search(&self, query: &str, k: usize) -> Vec<DocHit>;

    /// Devuelve offsets `(byte_start, byte_end)` de ocurrencias exactas de
    /// `needle` en el documento raw. `byte_end` es exclusivo. Sin stemming.
    fn locate(&self, doc_id: DocId, needle: &[u8], limit: usize) -> Vec<(u64, u64)>;

    /// Lista todos los doc_ids almacenados.
    fn list_docs(&self) -> Vec<DocId>;
}

/// Backend no-op (sin memoria documental).
/// Permite arrancar el nodo sin DocMemory configurada.
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
