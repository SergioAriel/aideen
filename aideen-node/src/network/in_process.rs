/// In-process channel for E2E tests without real I/O.
///
/// Simulates a bidirectional QUIC connection using std synchronous channels.
/// Both ends are created with `InProcessChannel::pair()`.
use std::sync::mpsc;

use aideen_core::protocol::NetMsg;

use super::NetChannel;

pub struct InProcessChannel {
    tx: mpsc::SyncSender<NetMsg>,
    rx: mpsc::Receiver<NetMsg>,
}

impl InProcessChannel {
    /// Creates a pair of connected channels: (client, server).
    /// What one end sends, the other receives.
    pub fn pair() -> (Self, Self) {
        let (tx_a, rx_b) = mpsc::sync_channel(64);
        let (tx_b, rx_a) = mpsc::sync_channel(64);
        (
            InProcessChannel { tx: tx_a, rx: rx_a },
            InProcessChannel { tx: tx_b, rx: rx_b },
        )
    }
}

impl NetChannel for InProcessChannel {
    fn send(&mut self, msg: NetMsg) -> Result<(), String> {
        self.tx.send(msg).map_err(|e| e.to_string())
    }

    fn recv(&mut self) -> Result<NetMsg, String> {
        self.rx.recv().map_err(|e| e.to_string())
    }
}
