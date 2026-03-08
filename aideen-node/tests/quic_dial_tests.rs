#![cfg(not(target_arch = "wasm32"))]

use aideen_node::peers::connector::FailureState;
use aideen_node::peers::types::NodeId;
use aideen_node::security::trust_store::{TrustDecision, TrustStore};

// ── Unit tests: TrustStore ────────────────────────────────────────────────────

#[test]
fn test_trust_store_tofu_first_time() {
    let mut store = TrustStore::new();
    let id: NodeId = [1u8; 32];
    let fp = [0xAAu8; 32];

    // Primera vez → TofuStored
    let r = store
        .verify_or_tofu(id, fp, None)
        .expect("first tofu must succeed");
    assert!(matches!(r, TrustDecision::TofuStored));

    // Segunda vez, mismo fp → Trusted
    let r2 = store
        .verify_or_tofu(id, fp, None)
        .expect("same fp must succeed");
    assert!(matches!(r2, TrustDecision::Trusted));
}

#[test]
fn test_trust_store_tofu_mismatch() {
    let mut store = TrustStore::new();
    let id: NodeId = [2u8; 32];
    let fp1 = [0x11u8; 32];
    let fp2 = [0x22u8; 32];

    store
        .verify_or_tofu(id, fp1, None)
        .expect("first tofu must succeed");

    // Segundo intento con fp distinto → Err (TOFU mismatch)
    let r = store.verify_or_tofu(id, fp2, None);
    assert!(r.is_err(), "fp change must be rejected");
}

#[test]
fn test_trust_store_pinned_ok() {
    let mut store = TrustStore::new();
    let id: NodeId = [3u8; 32];
    let fp = [0x33u8; 32];

    // Pinning explícito que coincide con el observado → TofuStored
    let r = store
        .verify_or_tofu(id, fp, Some(fp))
        .expect("matching pin must succeed");
    assert!(matches!(r, TrustDecision::TofuStored));
}

#[test]
fn test_trust_store_pinned_mismatch() {
    let mut store = TrustStore::new();
    let id: NodeId = [4u8; 32];
    let fp_observed = [0x44u8; 32];
    let fp_pinned = [0x55u8; 32];

    // Pinning que NO coincide con el observado → Err inmediato
    let r = store.verify_or_tofu(id, fp_observed, Some(fp_pinned));
    assert!(r.is_err(), "pin mismatch must be rejected");
}

// ── Unit test: CircuitBreaker ─────────────────────────────────────────────────

#[test]
fn test_circuit_breaker_opens_and_recovers() {
    let mut s = FailureState::default();

    // Sin fallos: puede intentar
    assert!(s.can_try(), "initially must be able to try");

    // Primer fallo: breaker se abre (backoff=1s)
    let node_id = [0u8; 32];
    s.record_failure(&node_id);
    assert_eq!(s.fail_count, 1);
    assert!(!s.can_try(), "after 1 failure breaker must be open");

    // Manipular open_until al pasado → simular expiración del TTL
    s.open_until = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    assert!(s.can_try(), "after TTL expired must be able to try again");

    // record_success → resetea
    s.record_success();
    assert_eq!(s.fail_count, 0);
    assert!(s.open_until.is_none());
    assert!(s.can_try(), "after success must be able to try");
}

// ── E2E test: QuicChannelFactory::dial() loopback ────────────────────────────

#[test]
fn test_quic_channel_factory_dial_loopback() {
    use std::net::SocketAddr;

    use rcgen::generate_simple_self_signed;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    use sha2::{Digest, Sha256};

    use aideen_core::protocol::NetMsg;
    use aideen_node::network::channel_factory::{ChannelFactory, QuicChannelFactory};

    // ── Construir cert auto-firmado (mismo patrón que pair_local) ─────────────
    let cert = generate_simple_self_signed(vec!["aideen-peer".to_string()]).unwrap();
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());
    let priv_key = PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());

    // Fingerprint esperado: sha256(cert_der)
    let expected_fp: [u8; 32] = Sha256::digest(cert_der.as_ref()).into();

    // ── Levantar servidor QUIC echo en hilo separado ──────────────────────────
    let (addr_tx, addr_rx) = std::sync::mpsc::sync_channel::<SocketAddr>(0);

    let cert_der_srv = cert_der.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let server_config =
                quinn::ServerConfig::with_single_cert(vec![cert_der_srv], priv_key).unwrap();
            let ep = quinn::Endpoint::server(
                server_config,
                "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            )
            .unwrap();
            let server_addr = ep.local_addr().unwrap();
            let _ = addr_tx.send(server_addr);

            // Aceptar 1 conexión y hacer echo de 1 mensaje
            if let Some(incoming) = ep.accept().await {
                let conn = incoming.await.unwrap();
                // Aceptar stream uni del cliente
                let mut recv = conn.accept_uni().await.unwrap();
                let mut len_buf = [0u8; 4];
                recv.read_exact(&mut len_buf).await.unwrap();
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; len];
                recv.read_exact(&mut buf).await.unwrap();
                let msg = NetMsg::decode(&buf).unwrap();

                // Responder con Pong en stream uni de salida
                let pong = match msg {
                    NetMsg::Ping => NetMsg::Pong,
                    _ => NetMsg::Pong,
                };
                let payload = pong.encode().unwrap();
                let mut send = conn.open_uni().await.unwrap();
                send.write_all(&(payload.len() as u32).to_le_bytes())
                    .await
                    .unwrap();
                send.write_all(&payload).await.unwrap();
                send.finish().unwrap();
                // Dar tiempo para que el cliente lea
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        });
    });

    let server_addr = addr_rx.recv().unwrap();

    // ── Dial con QuicChannelFactory ────────────────────────────────────────────
    let factory = QuicChannelFactory;
    let peer = aideen_node::peers::types::PeerEntry {
        node_id: [0u8; 32],
        endpoint: server_addr.to_string(), // "host:port" sin prefijo
        domains: vec![],
        bundle_version: 1,
        tls_fingerprint: None,
    };

    let dr = factory.dial(&peer).expect("dial must succeed");

    // Verificar fingerprint
    assert_eq!(
        dr.fingerprint, expected_fp,
        "DialResult.fingerprint must equal sha256(cert_der)"
    );

    // Enviar Ping, recibir Pong
    let mut ch = dr.channel;
    ch.send(NetMsg::Ping).expect("send Ping must succeed");
    let resp = ch.recv().expect("recv Pong must succeed");
    assert!(matches!(resp, NetMsg::Pong), "expected Pong, got {resp:?}");
}
