//! Dataset de texto char-level para training.
//!
//! Convierte texto arbitrario en pares (contexto, target) para
//! entrenar el pipeline AIDEEN: query → DEQ → H* → LmHead → token.

use nalgebra::DVector;

/// Dataset char-level: cada carácter único es un token.
pub struct TextDataset {
    /// Vocabulario: cada char es un token.
    pub vocab: Vec<char>,
    /// Texto tokenizado como indices [0, vocab_size).
    pub data: Vec<u32>,
}

impl TextDataset {
    /// Construye un dataset a partir de texto plano.
    /// Filtra caracteres repetidos para construir el vocabulario.
    pub fn from_str(text: &str) -> Self {
        let mut vocab: Vec<char> = Vec::new();
        for c in text.chars() {
            if !vocab.contains(&c) {
                vocab.push(c);
            }
        }
        vocab.sort();

        let data = text
            .chars()
            .filter_map(|c| vocab.iter().position(|&v| v == c).map(|i| i as u32))
            .collect();

        Self { vocab, data }
    }

    /// Dataset de ejemplo embebido para verificar que el sistema aprende.
    pub fn demo() -> Self {
        Self::from_str(
            "la inteligencia artificial distribuida razona en equilibrio \
             cada neurona artificial converge a un punto fijo estable \
             aideen es una red de neuronas artificiales distribuidas \
             el razonamiento emerge de la convergencia del equilibrio",
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Genera un par (query_vec, target_token) para training.
    /// `idx`: posición en el texto (0..data.len()-1)
    /// `d_r`: dimensión del vector de query.
    /// Retorna: (query como DVector, token target)
    pub fn sample(&self, idx: usize, d_r: usize) -> (DVector<f32>, u32) {
        let ctx_start = idx.saturating_sub(8);
        let ctx_end = idx.min(self.data.len() - 1);
        let target = self.data[(ctx_end + 1).min(self.data.len() - 1)];

        // Encode contexto como vector d_r usando hash simple
        let mut feats = vec![0.0f32; d_r];
        for (pos, &tok) in self.data[ctx_start..=ctx_end].iter().enumerate() {
            let slot = (tok as usize * 7 + pos * 13) % d_r;
            feats[slot] += 1.0;
            // Bigrama position-aware
            let slot2 = (tok as usize * 31 + pos * 57 + 1) % d_r;
            feats[slot2] += 0.3;
        }
        // Normalizar a tanh
        let query = DVector::from_vec(feats.into_iter().map(|x| x.tanh()).collect());
        (query, target)
    }

    /// Número de samples disponibles.
    pub fn len(&self) -> usize {
        self.data.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_dataset_has_content() {
        let ds = TextDataset::demo();
        assert!(ds.vocab_size() > 10, "vocab demasiado pequeño");
        assert!(ds.len() > 50, "dataset demasiado corto");
    }

    #[test]
    fn sample_produces_valid_target() {
        let ds = TextDataset::demo();
        let (query, target) = ds.sample(5, 512);
        assert_eq!(query.len(), 512);
        assert!((target as usize) < ds.vocab_size());
    }

    #[test]
    fn different_positions_different_queries() {
        let ds = TextDataset::demo();
        let (q1, _) = ds.sample(0, 512);
        let (q2, _) = ds.sample(10, 512);
        let diff: f32 = q1.iter().zip(q2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "queries distintas deben diferir");
    }
}
