#![cfg(target_arch = "wasm32")]

use aideen_core::doc_memory::{ChunkId, DocHit, DocId, DocMemory, DocMeta};

/// OPFS backend for DocMemory.
/// Compilable stub — correct shape, real implementation in the next sprint.
pub struct OpfsDocMemory;

impl OpfsDocMemory {
    pub async fn open(_agent_id: &str) -> Result<Self, String> {
        Ok(Self)
    }
}

impl DocMemory for OpfsDocMemory {
    fn add_document(&mut self, _meta: DocMeta, _bytes: Vec<u8>) -> Result<DocId, String> {
        Err("OpfsDocMemory: not yet implemented".into())
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
