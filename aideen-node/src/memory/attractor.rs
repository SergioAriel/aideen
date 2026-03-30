use aideen_core::memory::Memory;
use nalgebra::DVector;

/// Geometric attractor memory for h*.
///
/// Exact KNN by cosine similarity — WASM-safe, no external deps.
///
/// Stores `store_raw` (original h* for warm-start) and `store_normed`
/// (normalised h* for stable similarity). Doubles memory to 2x,
/// acceptable for < 10k attractors. If scaling: migrate to on-query
/// normalisation or compress `store_normed` to Vec<f32>.
pub struct AttractorMemory {
    store_raw: Vec<DVector<f32>>,
    store_normed: Vec<DVector<f32>>,
    dim: usize,
}

impl AttractorMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            store_raw: Vec::new(),
            store_normed: Vec::new(),
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.store_raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store_raw.is_empty()
    }

    fn normalize(v: &DVector<f32>) -> DVector<f32> {
        let n = v.norm();
        if n > 1e-8 {
            v / n
        } else {
            v.clone()
        }
    }

    /// Cosine similarity between two normalised vectors.
    /// Returns NEG_INFINITY if the result is not finite (anti-NaN).
    fn cosine(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
        let s = a.dot(b);
        if s.is_finite() {
            s
        } else {
            f32::NEG_INFINITY
        }
    }
}

impl Memory for AttractorMemory {
    fn write(&mut self, h: DVector<f32>) {
        assert_eq!(h.len(), self.dim, "AttractorMemory: dim mismatch en write");
        let normed = Self::normalize(&h);
        self.store_raw.push(h);
        self.store_normed.push(normed);
    }

    fn query(&self, query: &DVector<f32>, k: usize) -> Vec<DVector<f32>> {
        assert_eq!(
            query.len(),
            self.dim,
            "AttractorMemory: dim mismatch en query"
        );
        if self.store_normed.is_empty() || k == 0 {
            return vec![];
        }

        let q = Self::normalize(query);
        let n = self.store_normed.len();

        let mut scored: Vec<(f32, usize)> = self
            .store_normed
            .iter()
            .enumerate()
            .map(|(i, h)| (Self::cosine(&q, h), i))
            .collect();

        let take = k.min(n);

        // Top-k en O(n) + O(k log k):
        // select_nth_unstable_by(take-1) garantiza que scored[..take]
        // contiene los k mejores (no ordenados), luego ordenamos solo esos k.
        if take > 0 && take < n {
            scored.select_nth_unstable_by(take - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        scored[..take].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored[..take]
            .iter()
            .map(|(_, i)| self.store_raw[*i].clone())
            .collect()
    }
}
