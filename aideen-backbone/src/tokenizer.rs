//! Tokenizer char-level con embedding matrix trainable.
//!
//! Fase actual: char-level (cada carácter único es un token).
//! La clave no es BPE — es la **embedding matrix** que mapea tokens a D_R.
//! Sin embeddings, el DEQ recibe hashes sin estructura semántica.
//!
//! Fase futura: reemplazar char-level por BPE (crate `tokenizers`).
//!

use aideen_core::state::ArchitectureConfig;
use nalgebra::{DMatrix, DVector};

/// Tokenizer con vocabulario y embedding matrix trainable.
pub struct Tokenizer {
    pub config: ArchitectureConfig,
    /// Vocabulario: char → token ID.
    pub vocab: Vec<char>,
    /// Embedding matrix [vocab_size × D_R] — cada fila es el embedding de un token.
    pub embeddings: DMatrix<f32>,
}

impl Tokenizer {
    pub fn from_text(text: &str, config: ArchitectureConfig) -> Self {
        let mut vocab: Vec<char> = Vec::new();
        for c in text.chars() {
            if !vocab.contains(&c) {
                vocab.push(c);
            }
        }
        vocab.sort();

        let vocab_size = vocab.len();
        let d_r = config.d_r;
        let scale = (d_r as f32).sqrt().recip() * 0.5;
        let embeddings = DMatrix::from_fn(vocab_size, d_r, |i, j| {
            let v = ((i * d_r + j) as f32 * 1.6180339) % 1.0;
            (v - 0.5) * scale
        });

        Self {
            vocab,
            embeddings,
            config,
        }
    }

    pub fn new_empty(vocab_size: usize, config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        Self {
            vocab: Vec::with_capacity(vocab_size),
            embeddings: nalgebra::DMatrix::zeros(vocab_size, d_r),
            config,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.vocab.iter().position(|&v| v == c).map(|i| i as u32))
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.vocab.get(t as usize))
            .collect()
    }

    pub fn embed(&self, token_id: u32) -> DVector<f32> {
        self.embeddings.row(token_id as usize).transpose()
    }

    pub fn embed_context(&self, tokens: &[u32], max_ctx: usize) -> DVector<f32> {
        let ctx_start = tokens.len().saturating_sub(max_ctx);
        let ctx = &tokens[ctx_start..];

        let d_r = self.config.d_r;
        let mut result = DVector::zeros(d_r);
        for (pos, &tok) in ctx.iter().enumerate() {
            let emb = self.embed(tok);
            let pos_weight = (pos + 1) as f32 / ctx.len() as f32;
            result += emb * pos_weight;
        }
        let norm = result.norm();
        if norm > 1e-6 {
            result /= norm;
        }
        result
    }

    pub fn embed_sequence(&self, tokens: &[u32]) -> Vec<f32> {
        let d_r = self.config.d_r;
        let mut flat = Vec::with_capacity(tokens.len() * d_r);
        for &tok in tokens {
            let emb = self.embed(tok);
            flat.extend_from_slice(emb.as_slice());
        }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_roundtrip() {
        let config = ArchitectureConfig::default();
        let tok = Tokenizer::from_text("hola mundo", config);
        let encoded = tok.encode("hola");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "hola");
    }

    #[test]
    fn embed_has_correct_dim() {
        let config = ArchitectureConfig::default();
        let d_r = config.d_r;
        let tok = Tokenizer::from_text("test", config);
        let emb = tok.embed(0);
        assert_eq!(emb.len(), d_r);
    }

    #[test]
    fn different_tokens_different_embeddings() {
        let config = ArchitectureConfig::default();
        let tok = Tokenizer::from_text("ab", config);
        let ea = tok.embed(0);
        let eb = tok.embed(1);
        let diff: f32 = (&ea - &eb).norm();
        assert!(diff > 1e-6, "embeddings distintos para tokens distintos");
    }
}
