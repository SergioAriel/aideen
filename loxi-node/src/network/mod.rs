// Red P2P / Transporte
// Regla: NO puede importar `system`

use loxi_backbone::tensor::Tensor;

/// Contrato abstracto para el transporte de red (Chunked Prefill).
/// Evita que el `system` conozca sockets o WebRTC/libp2p directamente.
pub trait Transport {
    /// Inicia la conexión o el listening en la red.
    fn connect(&mut self) -> Result<(), String>;

    /// Envía un fragmento del tensor a la red (Micro-batching de ≈1.6MB).
    fn send_chunk(&mut self, chunk: &Tensor) -> Result<(), String>;

    /// Recibe un fragmento del tensor desde la red.
    fn receive_chunk(&mut self) -> Result<Tensor, String>;
}
