# Módulo local: `network` (Red P2P y Transporte)

## 📌 Responsabilidades
*   Manejar los sockets de comunicación (WebSocket, WebRTC, o libp2p).
*   Implementar la lógica del streaming y transporte P2P (Chunked Prefill de ≈1.6MB por envío).
*   Manejar the backpressure de la red.

## 🚫 Restricciones (Constitucionales)
*   **NO** toca el estado cognitivo ni el `State`.
*   **NO** llama directamente a funciones matemáticas o razonamiento.
*   **NO** importa dependencias del módulo `system` (el flujo va de `system` hacia `network`, no al revés).
