/// Canal in-process para tests E2E sin I/O real.
///
/// Simula una conexión QUIC bidireccional usando canales sincrónicos de std.
/// Ambos extremos se crean con `InProcessChannel::pair()`.
use std::sync::mpsc;

use aideen_core::protocol::NetMsg;

use super::NetChannel;

pub struct InProcessChannel {
    tx: mpsc::SyncSender<NetMsg>,
    rx: mpsc::Receiver<NetMsg>,
}

impl InProcessChannel {
    /// Crea un par de canales conectados: (client, server).
    /// Lo que un extremo envía lo recibe el otro.
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
