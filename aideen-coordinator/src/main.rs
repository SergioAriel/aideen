/// AIDEEN Coordinator — router + logger MVP.
///
/// Per-connection state:
///   Waiting → HelloReceived → Delegated → (can receive Discovery / receive Update)
///
/// Zero-Trust:
///   - Discovery received before Delegated → responds Error { code: 403 }
///   - All out-of-sequence messages → Error { code: 400 }
///
/// Framing identical to QuicChannel: u32 LE (length) + bincode(NetMsg).
use aideen_core::protocol::{AckKind, KeyDelegation, NetMsg, ParamId, QuantizedDelta};
use aideen_training::signer::{generate_master_keys, sign_key_delegation, sign_update};
use quinn::{Endpoint, RecvStream, SendStream, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::net::SocketAddr;
use std::sync::Arc;

// ── Framing (identical to QuicChannel) ─────────────────────────────────────

async fn send_msg(stream: &mut SendStream, msg: &NetMsg) -> Result<(), String> {
    let payload = msg.encode()?;
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len).await.map_err(|e| e.to_string())?;
    stream.write_all(&payload).await.map_err(|e| e.to_string())
}

async fn recv_msg(stream: &mut RecvStream) -> Result<NetMsg, String> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| e.to_string())?;
    NetMsg::decode(&buf)
}

// ── Session state ──────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum SessionState {
    Waiting,
    HelloReceived { node_id: [u8; 32] },
    Delegated { node_id: [u8; 32] },
}

// ── TLS Server ────────────────────────────────────────────────────────────

fn configure_server() -> (ServerConfig, Vec<u8>) {
    let cert = generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der_bytes = cert.cert.der().to_vec();
    let cert_der = CertificateDer::from(cert_der_bytes.clone());
    let priv_key = PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());
    let mut server_config =
        ServerConfig::with_single_cert(vec![cert_der], priv_key).expect("rustls server config");
    let transport = Arc::get_mut(&mut server_config.transport).unwrap();
    transport.max_concurrent_uni_streams(200_u32.into());
    (server_config, cert_der_bytes)
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AIDEEN Coordinator v1 (router + logger) ===");

    // Keys
    let (root_sk, _root_pk) = generate_master_keys();
    let (critic_sk, critic_pk) = generate_master_keys();

    // Pre-signed delegation (epoch 1, valid indefinitely)
    let delegation: KeyDelegation =
        sign_key_delegation(&root_sk, 1, critic_pk, 0, u64::MAX).expect("sign delegation");

    // Dummy update (in production this comes from the real Critic)
    let dummy_delta = QuantizedDelta {
        param: ParamId::W1,
        scale: 0.1,
        idx: vec![0],
        q: vec![1],
    };
    let dummy_update = sign_update(
        &critic_sk,
        1,
        "global".to_string(),
        0,
        [0u8; 32],
        [0u8; 32],
        [0u8; 32],
        vec![dummy_delta],
    )
    .expect("sign update");

    let delegation = Arc::new(NetMsg::Delegation(delegation));
    let update = Arc::new(NetMsg::Update(dummy_update));

    // Endpoint QUIC
    let addr: SocketAddr = "127.0.0.1:4433".parse()?;
    let (server_config, _cert) = configure_server();
    let endpoint = Endpoint::server(server_config, addr)?;
    println!("Coordinator listening on {}", addr);

    while let Some(incoming) = endpoint.accept().await {
        let delegation = Arc::clone(&delegation);
        let update = Arc::clone(&update);
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[coord] connection failed: {e}");
                    return;
                }
            };
            println!("[coord] client: {}", conn.remote_address());

            // Two uni streams: one for receiving (client→coord), one for sending (coord→client)
            let (tx, rx) = tokio::join!(conn.open_uni(), conn.accept_uni());
            let mut tx = match tx {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[coord] open_uni: {e}");
                    return;
                }
            };
            let mut rx = match rx {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[coord] accept_uni: {e}");
                    return;
                }
            };

            if let Err(e) = handle_session(&mut tx, &mut rx, &delegation, &update).await {
                eprintln!("[coord] session ended with error: {e}");
            }
        });
    }
    Ok(())
}

// ── Per-session state machine ─────────────────────────────────────────

async fn handle_session(
    tx: &mut SendStream,
    rx: &mut RecvStream,
    delegation: &NetMsg,
    update: &NetMsg,
) -> Result<(), String> {
    let mut state = SessionState::Waiting;

    loop {
        let msg = recv_msg(rx).await?;

        match (&state, msg) {
            // ── Hello ────────────────────────────────────────────────────
            (
                SessionState::Waiting,
                NetMsg::Hello {
                    node_id,
                    protocol,
                    bundle_version,
                    ..
                },
            ) => {
                println!(
                    "[coord] Hello node={:?} proto={} bv={}",
                    &node_id[..4],
                    protocol,
                    bundle_version
                );
                state = SessionState::HelloReceived { node_id };
                // Send Delegation
                send_msg(tx, delegation).await?;
            }

            // ── Ack de Delegation ────────────────────────────────────────
            (
                SessionState::HelloReceived { node_id },
                NetMsg::Ack {
                    kind: AckKind::Delegation,
                    version,
                    ok,
                },
            ) => {
                println!("[coord] Ack Delegation epoch={} ok={}", version, ok);
                let nid = *node_id;
                state = SessionState::Delegated { node_id: nid };
                // Send demo Update
                send_msg(tx, update).await?;
            }

            // ── Ack de Update ────────────────────────────────────────────
            (
                SessionState::Delegated { .. },
                NetMsg::Ack {
                    kind: AckKind::Update,
                    version,
                    ok,
                },
            ) => {
                println!("[coord] Ack Update v={} ok={}", version, ok);
                // Session complete for this demo cycle
                return Ok(());
            }

            // ── Discovery (only if already delegated) ─────────────────────
            (
                SessionState::Delegated { node_id },
                NetMsg::Discovery {
                    target_id,
                    q_total,
                    bundle_version,
                    ..
                },
            ) => {
                println!(
                    "[coord] Discovery node={:?} target={} q={:.3} bv={}",
                    &node_id[..4],
                    target_id,
                    q_total,
                    bundle_version
                );
                // Router: ACK and log. Does not evaluate yet (Critic pending).
                send_msg(
                    tx,
                    &NetMsg::Ack {
                        kind: AckKind::Discovery,
                        version: bundle_version,
                        ok: true,
                    },
                )
                .await?;
            }

            // ── Discovery without delegation = Zero-Trust reject ─────────────
            (_, NetMsg::Discovery { .. }) => {
                eprintln!("[coord] Discovery rejected: node not delegated");
                send_msg(
                    tx,
                    &NetMsg::Error {
                        code: 403,
                        msg: "Discovery requires prior Delegation".to_string(),
                    },
                )
                .await?;
                return Err("zero-trust: Discovery before Delegation".into());
            }

            // ── Ping ─────────────────────────────────────────────────────
            (_, NetMsg::Ping) => {
                send_msg(tx, &NetMsg::Pong).await?;
            }

            // ── Out-of-sequence message ────────────────────────────────
            (_, unexpected) => {
                eprintln!(
                    "[coord] unexpected message in state {:?}: {:?}",
                    state, unexpected
                );
                send_msg(
                    tx,
                    &NetMsg::Error {
                        code: 400,
                        msg: "unexpected message in current state".to_string(),
                    },
                )
                .await?;
                return Err("protocol error: unexpected message".into());
            }
        }
    }
}
