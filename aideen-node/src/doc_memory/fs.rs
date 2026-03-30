#![cfg(not(target_arch = "wasm32"))]

use aideen_core::doc_memory::{ChunkId, DocHit, DocId, DocMemory, DocMeta};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

// ── Tipos internos (solo en disco, no forman parte del contrato público) ──────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkOffset {
    chunk_id: ChunkId,
    byte_start: u64,
    byte_end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    chunk_id: ChunkId,
    tf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocRecord {
    doc_id: DocId,
    meta: DocMeta,
    chunk_offsets: Vec<ChunkOffset>,
}

// ── FsDocMemory ───────────────────────────────────────────────────────────────

/// Memoria documental persistente en disco (native only).
///
/// Layout:
///   <base_dir>/<agent_id>/docs/
///     manifest.bin          — bincode(Vec<DocRecord>)
///     raw/<doc_id>.bin      — bytes originales
///     chunks/<doc_id>.chunks — [u32 LE len][chunk_bytes]*
///     index/<doc_id>.index  — bincode(HashMap<String, Vec<Posting>>)
pub struct FsDocMemory {
    dir: PathBuf,
    manifest: Vec<DocRecord>,
    // Global inverted index in memory: token → [(doc_id, chunk_id, tf)]
    inverted: HashMap<String, Vec<(DocId, ChunkId, u32)>>,
    next_doc_id: DocId,
    chunk_size: usize, // bytes per chunk
    overlap: usize,    // overlap bytes between chunks
}

impl FsDocMemory {
    /// Opens or creates the store at `<base_dir>/<agent_id>/docs/`.
    /// Reconstructs the global index from disk at startup.
    pub fn open(base_dir: &str, agent_id: &str) -> Result<Self, String> {
        let dir = PathBuf::from(base_dir).join(agent_id).join("docs");
        std::fs::create_dir_all(dir.join("raw")).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(dir.join("chunks")).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(dir.join("index")).map_err(|e| e.to_string())?;

        let manifest_path = dir.join("manifest.bin");
        let manifest: Vec<DocRecord> = if manifest_path.exists() {
            let bytes = std::fs::read(&manifest_path).map_err(|e| e.to_string())?;
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            vec![]
        };

        let next_doc_id = manifest.iter().map(|r| r.doc_id).max().map_or(0, |m| m + 1);

        let mut slf = Self {
            dir,
            manifest,
            inverted: HashMap::new(),
            next_doc_id,
            chunk_size: 4096,
            overlap: 256,
        };
        slf.rebuild_inverted_index()?;
        Ok(slf)
    }

    /// Builder: configures chunk size and overlap.
    pub fn with_chunking(mut self, chunk_size: usize, overlap: usize) -> Self {
        self.chunk_size = chunk_size.max(512);
        self.overlap = overlap.min(self.chunk_size / 2);
        self
    }

    // ── Rutas ─────────────────────────────────────────────────────────────────

    fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.bin")
    }
    fn raw_path(&self, id: DocId) -> PathBuf {
        self.dir.join("raw").join(format!("{}.bin", id))
    }
    fn chunks_path(&self, id: DocId) -> PathBuf {
        self.dir.join("chunks").join(format!("{}.chunks", id))
    }
    fn index_path(&self, id: DocId) -> PathBuf {
        self.dir.join("index").join(format!("{}.index", id))
    }

    // ── Persistencia interna ──────────────────────────────────────────────────

    fn flush_manifest(&self) -> Result<(), String> {
        let bytes = bincode::serialize(&self.manifest).map_err(|e| e.to_string())?;
        std::fs::write(self.manifest_path(), bytes).map_err(|e| e.to_string())
    }

    fn rebuild_inverted_index(&mut self) -> Result<(), String> {
        self.inverted.clear();
        for rec in &self.manifest {
            let p = self.index_path(rec.doc_id);
            if !p.exists() {
                continue;
            }
            let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
            let local: HashMap<String, Vec<Posting>> =
                bincode::deserialize(&bytes).unwrap_or_default();
            for (tok, posts) in local {
                let entry = self.inverted.entry(tok).or_default();
                for post in posts {
                    entry.push((rec.doc_id, post.chunk_id, post.tf));
                }
            }
        }
        Ok(())
    }

    // ── Chunking ──────────────────────────────────────────────────────────────

    /// Splits `bytes` into chunks with overlap.
    /// Returns: Vec<(chunk_id, byte_start, byte_end_exclusive, chunk_bytes)>
    fn split_into_chunks(&self, bytes: &[u8]) -> Vec<(ChunkId, u64, u64, Vec<u8>)> {
        if bytes.is_empty() {
            return vec![];
        }
        let mut out = Vec::new();
        let mut start: usize = 0;
        let mut cid: ChunkId = 0;

        loop {
            let end = (start + self.chunk_size).min(bytes.len());
            out.push((cid, start as u64, end as u64, bytes[start..end].to_vec()));
            cid = cid.wrapping_add(1);
            if end == bytes.len() {
                break;
            }
            start = end.saturating_sub(self.overlap);
        }
        out
    }

    fn write_chunks_file(
        &self,
        doc_id: DocId,
        chunks: &[(ChunkId, u64, u64, Vec<u8>)],
    ) -> Result<(), String> {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(self.chunks_path(doc_id))
            .map_err(|e| e.to_string())?;
        for (_, _, _, payload) in chunks {
            f.write_all(&(payload.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(payload).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn read_chunk_from_file(&self, doc_id: DocId, chunk_id: ChunkId) -> Option<Vec<u8>> {
        let bytes = std::fs::read(self.chunks_path(doc_id)).ok()?;
        let mut cursor = 0usize;
        let mut cur_id: ChunkId = 0;
        while cursor + 4 <= bytes.len() {
            let len = u32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]) as usize;
            cursor += 4;
            if cursor + len > bytes.len() {
                break;
            }
            if cur_id == chunk_id {
                return Some(bytes[cursor..cursor + len].to_vec());
            }
            cursor += len;
            cur_id = cur_id.wrapping_add(1);
        }
        None
    }

    // ── Indexado ──────────────────────────────────────────────────────────────

    fn tokenize(text: &str) -> Vec<String> {
        text.chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .filter(|s| s.len() >= 2) // ignorar tokens de 1 char (ruido)
            .map(|s| s.to_string())
            .collect()
    }

    fn build_local_index(chunks: &[(ChunkId, u64, u64, Vec<u8>)]) -> HashMap<String, Vec<Posting>> {
        let mut acc: HashMap<String, HashMap<ChunkId, u32>> = HashMap::new();
        for (cid, _, _, payload) in chunks {
            let text = String::from_utf8_lossy(payload);
            for tok in Self::tokenize(&text) {
                *acc.entry(tok).or_default().entry(*cid).or_insert(0) += 1;
            }
        }
        acc.into_iter()
            .map(|(tok, map)| {
                let posts = map
                    .into_iter()
                    .map(|(cid, tf)| Posting { chunk_id: cid, tf })
                    .collect();
                (tok, posts)
            })
            .collect()
    }

    fn save_local_index(
        &self,
        doc_id: DocId,
        local: &HashMap<String, Vec<Posting>>,
    ) -> Result<(), String> {
        let bytes = bincode::serialize(local).map_err(|e| e.to_string())?;
        std::fs::write(self.index_path(doc_id), bytes).map_err(|e| e.to_string())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn preview(chunk: &[u8]) -> String {
        let s = String::from_utf8_lossy(chunk).to_string();
        if s.len() <= 240 {
            s
        } else {
            s[..240].to_string()
        }
    }

    fn record_for(&self, doc_id: DocId) -> Option<&DocRecord> {
        self.manifest.iter().find(|r| r.doc_id == doc_id)
    }
}

// ── DocMemory impl ────────────────────────────────────────────────────────────

impl DocMemory for FsDocMemory {
    fn add_document(&mut self, mut meta: DocMeta, bytes: Vec<u8>) -> Result<DocId, String> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;
        meta.len_bytes = bytes.len() as u64;

        // 1) Persistir raw
        std::fs::write(self.raw_path(doc_id), &bytes).map_err(|e| e.to_string())?;

        // 2) Chunk + persistir chunks file
        let chunks = self.split_into_chunks(&bytes);
        self.write_chunks_file(doc_id, &chunks)?;

        // 3) Extraer offsets para manifest
        let chunk_offsets = chunks
            .iter()
            .map(|(cid, bs, be, _)| ChunkOffset {
                chunk_id: *cid,
                byte_start: *bs,
                byte_end: *be,
            })
            .collect();

        // 4) Local index → disk + merge into global in-memory index
        let local_index = Self::build_local_index(&chunks);
        self.save_local_index(doc_id, &local_index)?;
        for (tok, posts) in local_index {
            let entry = self.inverted.entry(tok).or_default();
            for post in posts {
                entry.push((doc_id, post.chunk_id, post.tf));
            }
        }

        // 5) Update manifest + flush
        self.manifest.push(DocRecord {
            doc_id,
            meta,
            chunk_offsets,
        });
        self.flush_manifest()?;

        Ok(doc_id)
    }

    fn get_meta(&self, doc_id: DocId) -> Option<DocMeta> {
        self.record_for(doc_id).map(|r| r.meta.clone())
    }

    fn get_chunk(&self, doc_id: DocId, chunk_id: ChunkId) -> Option<Vec<u8>> {
        self.read_chunk_from_file(doc_id, chunk_id)
    }

    fn search(&self, query: &str, k: usize) -> Vec<DocHit> {
        if k == 0 {
            return vec![];
        }
        let toks = Self::tokenize(query);
        if toks.is_empty() {
            return vec![];
        }

        // Accumulate TF score per (doc_id, chunk_id)
        let mut scores: HashMap<(DocId, ChunkId), f32> = HashMap::new();
        for tok in toks {
            if let Some(posts) = self.inverted.get(&tok) {
                for (doc_id, chunk_id, tf) in posts {
                    *scores.entry((*doc_id, *chunk_id)).or_insert(0.0) += *tf as f32;
                }
            }
        }

        // Top-k por score
        let mut items: Vec<_> = scores.into_iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        items.truncate(k);

        items
            .into_iter()
            .map(|((doc_id, chunk_id), score)| {
                let (byte_start, byte_end) = self
                    .record_for(doc_id)
                    .and_then(|r| r.chunk_offsets.iter().find(|o| o.chunk_id == chunk_id))
                    .map(|o| (o.byte_start, o.byte_end))
                    .unwrap_or((0, 0));
                let preview = self
                    .read_chunk_from_file(doc_id, chunk_id)
                    .map(|c| Self::preview(&c))
                    .unwrap_or_default();
                DocHit {
                    doc_id,
                    chunk_id,
                    score,
                    byte_start,
                    byte_end,
                    preview,
                }
            })
            .collect()
    }

    fn locate(&self, doc_id: DocId, needle: &[u8], limit: usize) -> Vec<(u64, u64)> {
        if needle.is_empty() || limit == 0 {
            return vec![];
        }
        let Ok(bytes) = std::fs::read(self.raw_path(doc_id)) else {
            return vec![];
        };

        let mut out = Vec::new();
        let mut i = 0usize;
        while i + needle.len() <= bytes.len() {
            if bytes[i..i + needle.len()] == *needle {
                out.push((i as u64, (i + needle.len()) as u64));
                if out.len() >= limit {
                    break;
                }
                i += needle.len(); // sin overlapping matches
            } else {
                i += 1;
            }
        }
        out
    }

    fn list_docs(&self) -> Vec<DocId> {
        self.manifest.iter().map(|r| r.doc_id).collect()
    }
}
