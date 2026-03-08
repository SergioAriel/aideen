// Red P2P / Transporte
// Regla: NO puede importar `system`

use aideen_core::protocol::NetMsg;

/// Canal de mensajes NetMsg — wire-format canónico de AIDEEN.
///
/// Toda comunicación de protocolo (handshake, discovery, updates, expert tasks)
/// usa este canal. tick() no toca este canal — los mensajes se emiten afuera del loop DEQ.
pub trait NetChannel {
    fn send(&mut self, msg: NetMsg) -> Result<(), String>;
    fn recv(&mut self) -> Result<NetMsg, String>;
}

pub mod channel_factory;
pub mod in_process;
pub mod quic_channel;
