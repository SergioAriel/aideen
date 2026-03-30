use crate::network::NetChannel;
use crate::peers::types::PeerEntry;

/// Result of a successful dial: ready channel + SHA-256 fingerprint of the peer's leaf cert.
pub struct DialResult {
    pub channel: Box<dyn NetChannel>,
    pub fingerprint: [u8; 32],
}

/// Channel factory from a PeerEntry. Implemented by QuicChannelFactory (and mocks in tests).
pub trait ChannelFactory: Send + Sync {
    fn dial(&self, peer: &PeerEntry) -> Result<DialResult, String>;
}

/// Real implementation: QUIC dial with fingerprint capture.
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

/// Factory that always returns an error — default in NodeRunner::new() for tests without real dial.
pub struct NullChannelFactory;

impl ChannelFactory for NullChannelFactory {
    fn dial(&self, _: &PeerEntry) -> Result<DialResult, String> {
        Err("NullChannelFactory: no dial configured".into())
    }
}
