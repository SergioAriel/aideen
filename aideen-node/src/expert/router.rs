/// Selecciona peers y pesos α (normalizados, Σα=1) para consulta de expertos.
pub trait Router: Send {
    fn select(&self, h_k: &[f32], n_peers: usize) -> Vec<(usize, f32)>;
}

/// Pesos uniformes sobre los primeros K peers.
pub struct UniformRouter {
    pub k: usize,
}

impl Router for UniformRouter {
    fn select(&self, _h_k: &[f32], n_peers: usize) -> Vec<(usize, f32)> {
        let k = self.k.min(n_peers);
        if k == 0 {
            return vec![];
        }
        let alpha = 1.0 / k as f32;
        (0..k).map(|i| (i, alpha)).collect()
    }
}
