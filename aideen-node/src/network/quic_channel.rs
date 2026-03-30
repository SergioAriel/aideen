/// QuicChannel: implements NetChannel over quinn QUIC.
///
/// Canonical framing: u32 LE (length) + bincode(NetMsg).
/// Internally uses tokio::sync::mpsc for sync↔async bridging.
///
/// Architecture:
/// - Persistent global runtime (OnceLock) to avoid losing tasks.
/// - pair_local() connects, signals readiness, and spawns I/O tasks.
/// - dial() opens a real connection to "host:port" (or "quic://host:port"), captures fingerprint.
/// - open_uni for outbound / accept_uni for inbound: either side can write first.
use std::sync::{Arc, OnceLock};

use aideen_core::protocol::NetMsg;

use super::NetChannel;

pub struct QuicChannel {
    tx: tokio::sync::mpsc::Sender<NetMsg>,
    rx: tokio::sync::mpsc::Receiver<NetMsg>,
}

impl QuicChannel {
    fn new(tx: tokio::sync::mpsc::Sender<NetMsg>, rx: tokio::sync::mpsc::Receiver<NetMsg>) -> Self {
        Self { tx, rx }
    }
}

impl NetChannel for QuicChannel {
    fn send(&mut self, msg: NetMsg) -> Result<(), String> {
        self.tx.blocking_send(msg).map_err(|e| e.to_string())
    }

    fn recv(&mut self) -> Result<NetMsg, String> {
        self.rx
            .blocking_recv()
            .ok_or_else(|| "channel closed".to_string())
    }
}

// ── Framing helpers ───────────────────────────────────────────────────────

/// Escribe NetMsg con framing: u32 LE (longitud) + payload bincode.
async fn write_framed(send: &mut quinn::SendStream, msg: &NetMsg) -> Result<(), String> {
    let payload = msg.encode()?;
    let len = (payload.len() as u32).to_le_bytes();
    send.write_all(&len).await.map_err(|e| e.to_string())?;
    send.write_all(&payload).await.map_err(|e| e.to_string())
}

/// Lee NetMsg con framing: u32 LE (longitud) + payload bincode.
async fn read_framed(recv: &mut quinn::RecvStream) -> Result<NetMsg, String> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf)
        .await
        .map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await.map_err(|e| e.to_string())?;
    NetMsg::decode(&buf)
}

// ── Runtime global ────────────────────────────────────────────────────────

/// Shared tokio runtime — lives until the end of the process.
/// Necessary so that QUIC tasks survive pair_local().
static QUIC_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn quic_rt() -> &'static tokio::runtime::Runtime {
    QUIC_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap()
    })
}

// ── Test helpers ──────────────────────────────────────────────────────────

impl QuicChannel {
    /// Creates a (client, server) pair connected over local QUIC with ephemeral ports.
    /// Generates a self-signed certificate. For tests only.
    /// Blocks until the QUIC connection is established.
    pub fn pair_local() -> (Self, Self) {
        use std::net::SocketAddr;
        use std::sync::Arc;

        use rcgen::generate_simple_self_signed;
        use rustls::pki_types::{CertificateDer, PrivateKeyDer};

        let cert = generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_der = CertificateDer::from(cert.cert.der().to_vec());
        let priv_key = PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());

        // Sync↔async channels
        let (c_app_tx, c_net_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);
        let (c_net_tx, c_app_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);
        let (s_app_tx, s_net_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);
        let (s_net_tx, s_app_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);

        // Readiness signal (synchronous: ready when connection established)
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<()>(0);

        let cert_der_srv = cert_der.clone();
        let cert_der_cli = cert_der.clone();

        quic_rt().spawn(async move {
            // ── Server endpoint ──────────────────────────────────────────
            let server_config =
                quinn::ServerConfig::with_single_cert(vec![cert_der_srv], priv_key).unwrap();
            let server_ep = quinn::Endpoint::server(
                server_config,
                "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            )
            .unwrap();
            let server_addr = server_ep.local_addr().unwrap();

            // ── Client config — trusts solo este cert ────────────────────
            let mut root_store = rustls::RootCertStore::empty();
            root_store.add(cert_der_cli).unwrap();
            let rustls_cli = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            let quic_cli = quinn::crypto::rustls::QuicClientConfig::try_from(rustls_cli).unwrap();
            let client_config = quinn::ClientConfig::new(Arc::new(quic_cli));

            let client_ep =
                quinn::Endpoint::client("127.0.0.1:0".parse::<SocketAddr>().unwrap()).unwrap();

            // ── Conectar concurrentemente ────────────────────────────────
            let (client_conn, server_conn) = tokio::join!(
                async {
                    client_ep
                        .connect_with(client_config, server_addr, "localhost")
                        .unwrap()
                        .await
                        .unwrap()
                },
                async {
                    let incoming = server_ep.accept().await.unwrap();
                    incoming.await.unwrap()
                }
            );

            // ── Signal readiness AFTER connection ─────────────────────────
            // Each side opens ITS OWN outbound stream (open_uni) and accepts
            // the other's (accept_uni). This way either can write first
            // without blocking the other on accept.
            let _ = ready_tx.send(());

            // ── I/O loops: 2 uni streams independientes ──────────────────
            //
            //   C→S: client opens uni stream → server accepts
            //   S→C: server opens uni stream → client accepts
            //
            // Cada write loop abre su stream lazily cuando llega el 1er msg.
            // Cada read loop acepta el stream del peer cuando éste escribe.
            let _client_ep = client_ep;
            let _server_ep = server_ep;

            // Clonar conexiones: cada una se usa en 2 closures (writer + reader)
            let client_conn_r = client_conn.clone();
            let server_conn_r = server_conn.clone();

            tokio::join!(
                // Cliente writer: abre stream uni C→S, escribe mensajes
                async move {
                    let mut send = client_conn.open_uni().await.unwrap();
                    let mut rx = c_net_rx;
                    while let Some(msg) = rx.recv().await {
                        if write_framed(&mut send, &msg).await.is_err() {
                            break;
                        }
                    }
                },
                // Cliente reader: acepta stream uni S→C cuando servidor escribe
                async move {
                    let mut recv = client_conn_r.accept_uni().await.unwrap();
                    loop {
                        match read_framed(&mut recv).await {
                            Ok(msg) => {
                                if c_net_tx.send(msg).await.is_err() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                },
                // Servidor writer: abre stream uni S→C, escribe mensajes
                async move {
                    let mut send = server_conn.open_uni().await.unwrap();
                    let mut rx = s_net_rx;
                    while let Some(msg) = rx.recv().await {
                        if write_framed(&mut send, &msg).await.is_err() {
                            break;
                        }
                    }
                },
                // Servidor reader: acepta stream uni C→S cuando cliente escribe
                async move {
                    let mut recv = server_conn_r.accept_uni().await.unwrap();
                    loop {
                        match read_framed(&mut recv).await {
                            Ok(msg) => {
                                if s_net_tx.send(msg).await.is_err() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                },
            );
        });

        ready_rx.recv().unwrap();

        (
            QuicChannel::new(c_app_tx, c_app_rx),
            QuicChannel::new(s_app_tx, s_app_rx),
        )
    }
}

// ── FingerprintCapture ────────────────────────────────────────────────────────

/// Custom TLS verifier: accepts any cert and captures its SHA-256 fingerprint.
/// TOFU/pinning validation occurs after the handshake in TrustStore.
/// verify_tls12/13_signature → assertion() — security is post-handshake.
#[derive(Debug)]
struct FingerprintCapture {
    fingerprint: Arc<OnceLock<[u8; 32]>>,
}

impl rustls::client::danger::ServerCertVerifier for FingerprintCapture {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        use sha2::{Digest, Sha256};
        let fp: [u8; 32] = Sha256::digest(end_entity.as_ref()).into();
        let _ = self.fingerprint.set(fp);
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _msg: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _msg: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA256,
        ]
    }
}

// ── QuicChannel::dial() ───────────────────────────────────────────────────────

impl QuicChannel {
    /// Conecta a un endpoint real ("host:port" o "quic://host:port"), captura fingerprint.
    /// Blocks until the QUIC handshake completes.
    pub fn dial(endpoint: &str) -> Result<(Self, [u8; 32]), String> {
        // Ajuste #1: soportar "quic://host:port" y "host:port"
        let endpoint = endpoint.strip_prefix("quic://").unwrap_or(endpoint);

        let fp_cell: Arc<OnceLock<[u8; 32]>> = Arc::new(OnceLock::new());
        let verifier = Arc::new(FingerprintCapture {
            fingerprint: fp_cell.clone(),
        });

        // Build ClientConfig with custom verifier (requires dangerous_configuration feature)
        let rustls_cli = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        let quic_cli = quinn::crypto::rustls::QuicClientConfig::try_from(rustls_cli)
            .map_err(|e| e.to_string())?;
        let client_config = quinn::ClientConfig::new(Arc::new(quic_cli));

        let addr: std::net::SocketAddr = endpoint
            .parse()
            .map_err(|e: std::net::AddrParseError| e.to_string())?;

        let (app_tx, net_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);
        let (net_tx, app_rx) = tokio::sync::mpsc::channel::<NetMsg>(64);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(0);

        quic_rt().spawn(async move {
            // Ajuste #4: .await directo dentro del spawn — mismo patrón que pair_local()
            let client_ep =
                match quinn::Endpoint::client("0.0.0.0:0".parse::<std::net::SocketAddr>().unwrap())
                {
                    Ok(ep) => ep,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };

            let connecting = match client_ep.connect_with(client_config, addr, "aideen-peer") {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };

            let conn = match connecting.await {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };

            let _ = ready_tx.send(Ok(()));

            // I/O loops: uni stream saliente (open_uni) + uni stream entrante (accept_uni)
            let conn_r = conn.clone();
            tokio::join!(
                async move {
                    let mut send = conn.open_uni().await.unwrap();
                    let mut rx = net_rx;
                    while let Some(msg) = rx.recv().await {
                        if write_framed(&mut send, &msg).await.is_err() {
                            break;
                        }
                    }
                },
                async move {
                    let mut recv = conn_r.accept_uni().await.unwrap();
                    loop {
                        match read_framed(&mut recv).await {
                            Ok(msg) => {
                                if net_tx.send(msg).await.is_err() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                },
            );
        });

        ready_rx.recv().unwrap()?;

        let fp = *fp_cell.get().ok_or("fingerprint not captured")?;
        Ok((QuicChannel::new(app_tx, app_rx), fp))
    }
}
