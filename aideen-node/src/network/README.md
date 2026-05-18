# Local module: `network` (P2P Network and Transport)

## 📌 Responsibilities
*   Handle the communication sockets (WebSocket, WebRTC, or libp2p).
*   Implement the streaming and P2P transport logic (Chunked Prefill of ≈1.6MB per send).
*   Handle the network backpressure.

## 🚫 Restrictions (Constitutional)
*   Does **NOT** touch the cognitive state nor the `State`.
*   Does **NOT** directly call mathematical or reasoning functions.
*   Does **NOT** import dependencies from the `system` module (the flow goes from `system` to `network`, not the other way around).
