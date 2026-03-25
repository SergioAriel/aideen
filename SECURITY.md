# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AIDEEN, please report it responsibly.

**Email:** security@aideen.dev (or open a private security advisory on GitHub)

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Affected component (core, backbone, block, node, coordinator)
- Potential impact

**Response timeline:**
- Acknowledgment within 48 hours
- Assessment within 7 days
- Fix or mitigation within 30 days for critical issues

## Scope

AIDEEN's security-relevant components include:
- **aideen-core:** Sealed protocol, Ed25519 signatures, cryptographic state verification
- **aideen-node:** QUIC/WebTransport networking, TLS via rustls
- **aideen-coordinator:** Key delegation, governance ledger, update authorization
- **GPU shaders:** Buffer bounds, memory access patterns in WGSL compute shaders

## Cryptographic Dependencies

- `ed25519-dalek 2` for signatures
- `sha2` for hashing
- `rustls 0.23` for TLS
- `quinn 0.11` for QUIC transport

All dependencies are from crates.io and regularly audited by the Rust security community.
