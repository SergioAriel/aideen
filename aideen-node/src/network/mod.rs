// P2P Network / Transport
// Rule: MUST NOT import `system`

use aideen_core::protocol::NetMsg;

/// NetMsg message channel — AIDEEN canonical wire-format.
///
/// All protocol communication (handshake, discovery, updates, expert tasks)
/// uses this channel. tick() does not touch this channel — messages are emitted outside the DEQ loop.
pub trait NetChannel {
    fn send(&mut self, msg: NetMsg) -> Result<(), String>;
    fn recv(&mut self) -> Result<NetMsg, String>;
}

pub mod channel_factory;
pub mod in_process;
pub mod quic_channel;
