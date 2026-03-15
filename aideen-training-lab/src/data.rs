/// Data loader for training sequences.
///
/// Stores a flat token corpus and yields random contiguous (input, target)
/// pairs where target is the input shifted by one position.
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::Path;

pub struct DataLoader {
    pub tokens: Vec<u32>,
    pub ctx_len: usize,
    rng: StdRng,
}

impl DataLoader {
    /// Create a new DataLoader from an in-memory token vector.
    ///
    /// * `tokens` - the full token corpus (must have length > ctx_len)
    /// * `ctx_len` - context window length
    /// * `seed` - seed for the random number generator
    pub fn new(tokens: Vec<u32>, ctx_len: usize, seed: u64) -> Self {
        assert!(
            tokens.len() > ctx_len,
            "Token corpus length ({}) must be greater than ctx_len ({})",
            tokens.len(),
            ctx_len
        );
        Self {
            tokens,
            ctx_len,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Load tokens from a binary file containing little-endian u32 values.
    ///
    /// Returns `Err(String)` if the file cannot be read or is empty / too short.
    pub fn from_file(path: impl AsRef<Path>, ctx_len: usize, seed: u64) -> Result<Self, String> {
        let bytes = fs::read(path.as_ref())
            .map_err(|e| format!("Failed to read file {:?}: {}", path.as_ref(), e))?;

        if bytes.len() % 4 != 0 {
            return Err(format!(
                "File size ({}) is not a multiple of 4 bytes",
                bytes.len()
            ));
        }

        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        if tokens.len() <= ctx_len {
            return Err(format!(
                "Token corpus length ({}) must be greater than ctx_len ({})",
                tokens.len(),
                ctx_len
            ));
        }

        Ok(Self {
            tokens,
            ctx_len,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Number of tokens in the corpus.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the corpus is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Sample a random contiguous (input, target) pair.
    ///
    /// A random starting index `i` is chosen such that
    /// `tokens[i .. i + ctx_len + 1]` is valid. Then:
    /// - `input  = tokens[i   .. i + ctx_len]`
    /// - `target = tokens[i+1 .. i + ctx_len + 1]`
    pub fn next_batch(&mut self) -> (Vec<u32>, Vec<u32>) {
        // We need a window of ctx_len + 1 tokens.
        // Valid start indices: 0 ..= tokens.len() - ctx_len - 1
        let max_start = self.tokens.len() - self.ctx_len - 1;
        let start = self.rng.gen_range(0..=max_start);

        let input = self.tokens[start..start + self.ctx_len].to_vec();
        let target = self.tokens[start + 1..start + self.ctx_len + 1].to_vec();

        (input, target)
    }

    /// Split a token vector into train and validation sets.
    ///
    /// `ratio` is the fraction that goes to the first (train) set.
    /// For example, `ratio = 0.9` means 90 % train, 10 % validation.
    pub fn split(tokens: &[u32], ratio: f64) -> (Vec<u32>, Vec<u32>) {
        assert!(
            (0.0..=1.0).contains(&ratio),
            "ratio must be between 0.0 and 1.0, got {}",
            ratio
        );
        let split_idx = (tokens.len() as f64 * ratio).round() as usize;
        let train = tokens[..split_idx].to_vec();
        let val = tokens[split_idx..].to_vec();
        (train, val)
    }
}
