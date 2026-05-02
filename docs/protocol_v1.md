# AIDEEN Network Protocol v1

> **Status**: Frozen — compatible con `aideen-core >= 0.3`
> **Wire version**: `Hello.protocol = 1`

---

## 1. Framing

Todos los mensajes usan el mismo framing binario, tanto en QUIC como en WebTransport:

```
┌──────────────┬────────────────────────────────────┐
│  length (4B) │  payload (bincode)                  │
│  u32 LE      │  NetMsg (serde / bincode)            │
└──────────────┴────────────────────────────────────┘
```

- **4 bytes**: longitud del payload en little-endian (`u32`).
- **Payload**: `bincode::serialize(&NetMsg)` — no versioning propio, el discriminante de enum actúa como type tag.
- Sin padding. Sin delimitador final.

### Transporte

| Transporte | Stream     | Dirección        |
|------------|------------|------------------|
| QUIC       | uni stream | C→S: client abre; S→C: server abre |
| WebTransport (futuro) | unidirectional stream | igual |
| InProcess (tests)  | `sync_channel(64)` | — |

Dos uni streams independientes eliminan el deadlock de inicialización: cada parte abre su propio stream saliente y acepta el del otro lado, independientemente de quién escribe primero.

---

## 2. Enum `NetMsg`

### `Hello`
```
Hello {
    node_id:        [u8; 32],   // Clave pública efímera del nodo
    protocol:       u32,        // = 1 para esta versión
    bundle_version: u64,        // Bundle de modelos instalado
    bundle_hash:    [u8; 32],   // SHA-256 del bundle (alineación topológica)
}
```
Primer mensaje del cliente al abrir sesión. El servidor responde con `Delegation`.

### `Delegation(KeyDelegation)`
```
KeyDelegation {
    epoch:               u64,
    critic_pk:           [u8; 32],
    valid_from_unix:     u64,
    valid_to_unix:       u64,
    signature_by_root:   Vec<u8>,   // ed25519 sobre signing_bytes()
}
```
El coordinator delega la llave operativa del Critic actual. La firma se verifica contra la `root_pk` embebida en el nodo. Solo se acepta si `epoch > current_epoch` (anti-rollback).

### `Ack`
```
Ack {
    kind:    AckKind,   // Delegation | Update | Discovery
    version: u64,       // epoch (Delegation) o version (Update/Discovery)
    ok:      bool,
}
```
Confirmación tipificada. El campo `kind` permite al coordinator verificar que el cliente confirmó el mensaje correcto sin ambigüedad.

### `Update(SignedUpdate)`
```
SignedUpdate {
    version:          u64,
    target_id:        String,      // expert que recibe el delta
    bundle_version:   u64,
    bundle_hash:      [u8; 32],
    base_model_hash:  [u8; 32],    // snapshot sobre el cual aplica el delta
    prev_update_hash: [u8; 32],    // encadenado anti-fork
    payload:          Vec<u8>,     // zstd( bincode( Vec<QuantizedDelta> ) )
    signature:        Vec<u8>,     // ed25519 sobre signing_bytes()
}
```
El cliente verifica firma + `base_model_hash` + anti-replay (`version > last_seen_version`) antes de aplicar.

### `Discovery`
```
Discovery {
    node_id:        [u8; 32],
    target_id:      String,
    q_total:        f32,
    iters:          u32,
    stop:           u8,          // 0 = Q_MIN_WRITE gate, 1 = Epsilon gate
    h_star_hash:    [u8; 32],    // SHA-256 de h* serializado en LE
    context_hash:   [u8; 32],    // SHA-256 del s_context
    bundle_version: u64,
}
```
Señal de calidad enviada por el nodo cuando `Q >= Q_MIN_LEARN`. **No contiene `h*` completo** — solo hashes para correlación, dedup y anti-spam. El Architect decide si solicita más información al nodo.

### `ExpertTask` / `ExpertResult`
```
ExpertTask {
    task_id:        [u8; 16],
    target_id:      String,
    s_r:            Vec<f32>,    // subespacio de razonamiento (~8KB)
    bundle_version: u64,
}

ExpertResult {
    task_id:  [u8; 16],
    target_id: String,
    delta:    Vec<f32>,    // Δ = h_next − h (~8KB)
    q_total:  f32,
    iters:    u32,
    stop:     u8,
}
```
Routing de tareas de inferencia peer-to-peer entre nodos expertos.

### `Metrics`
```
Metrics { q_total: f32, iters: u32, stop: u8 }
```
Telemetría ligera, no genera Discovery. Se usa para monitoreo sin activar el ciclo de aprendizaje.

### `Error`
```
Error { code: u32, msg: String }
```
Códigos:
- `400` — mensaje inesperado en el estado actual de sesión.
- `403` — Discovery recibido sin Delegation previa (Zero-Trust).

### `Ping` / `Pong`
Heartbeat. El coordinator responde `Pong` a cualquier `Ping` en cualquier estado.

---

## 3. Máquina de estados del Coordinator

```
        Hello
[Waiting] ──────────────► [HelloReceived]
                                │
                           Ack(Delegation)
                                │
                                ▼
                          [Delegated]
                           │        │
                    Discovery       Ack(Update)
                     → Ack       → sesión completa
```

Transiciones fuera de secuencia generan `Error { code: 400 }` y cierran la sesión.

---

## 4. Invariantes del protocolo

1. **No `h*` en wire**: `Discovery` solo transporta `h_star_hash` (SHA-256). El vector completo nunca sale del nodo por esta ruta.

2. **Anti-replay en Delegation**: `epoch` debe ser estrictamente mayor que `current_epoch`. El nodo rechaza delegaciones con epoch igual o menor.

3. **Anti-replay en Update**: `version` debe ser mayor que la última versión aplicada para ese `target_id`. El `ReplayGuard` mantiene el mapa por target.

4. **Encadenado de updates**: `prev_update_hash` crea una cadena que el Critic puede verificar para detectar forks o reordenamientos maliciosos.

5. **Zero-Trust**: `Discovery` antes de `Delegation` → `Error { code: 403 }`. El coordinator no acepta inteligencia sin autorización criptográfica previa.

6. **Firma doble**: Delegation (firmado por root_sk) + Update (firmado por critic_sk delegado). El nodo verifica ambas cadenas antes de aplicar cambios.

---

## 5. Compatibilidad futura con WebTransport

WebTransport sobre HTTP/3 usa el mismo mecanismo QUIC subyacente. La migración requiere:

- Reemplazar `quinn::SendStream` / `RecvStream` por `web_transport::SendStream` / `RecvStream`.
- El framing (u32 LE + bincode) permanece **idéntico** — no hay cambio de protocolo.
- El discriminante `Hello.protocol = 1` permite negociar versiones futuras sin romper compatibilidad.
- En WASM, WebTransport reemplaza QUIC nativo; el `NetMsg` enum y el framing son los mismos.

---

## 6. Registro de versiones

| Versión wire | Cambio principal |
|---|---|
| 1 | Inicial: Hello + Delegation + Update + Ack(AckKind) + Discovery + Error |

---

## Appendix A — Distributed inference design note

`ExpertTask` / `ExpertResult` are the v1 MVP wire shape for expert deltas. The current architectural target for distributed inference is broader: experts should preferably return structured memory/context/evidence patches that are locally gated, integrated, and refined before the LMHead decodes.

See [docs/vision/distributed_inference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/distributed_inference.md) for the current design direction. This appendix does not change wire version `1`; richer `ExpertQuery` / `ExpertPatch` messages should be introduced only through an explicit future protocol version.
