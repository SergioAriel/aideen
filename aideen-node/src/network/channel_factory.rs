use crate::network::NetChannel;
use crate::peers::types::PeerEntry;

/// Resultado de un dial exitoso: canal listo + fingerprint SHA-256 del cert leaf del peer.
pub struct DialResult {
    pub channel: Box<dyn NetChannel>,
    pub fingerprint: [u8; 32],
}

/// Fábrica de canales desde un PeerEntry. Implementado por QuicChannelFactory (y mocks en tests).
pub trait ChannelFactory: Send + Sync {
    fn dial(&self, peer: &PeerEntry) -> Result<DialResult, String>;
}

/// Implementación real: dial QUIC con fingerprint capture.
pub struct QuicChannelFactory;

impl ChannelFactory for QuicChannelFactory {
    fn dial(&self, peer: &PeerEntry) -> Result<DialResult, String> {
        let (channel, fingerprint) =
            crate::network::quic_channel::QuicChannel::dial(&peer.endpoint)?;
        Ok(DialResult {
            channel: Box::new(channel),
            fingerprint,
        })
    }
}

/// Factory que siempre retorna error — default en NodeRunner::new() para tests sin dial real.
pub struct NullChannelFactory;

impl ChannelFactory for NullChannelFactory {
    fn dial(&self, _: &PeerEntry) -> Result<DialResult, String> {
        Err("NullChannelFactory: no dial configured".into())
    }
}
