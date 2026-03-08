# aideen-ai

Repositorio privado — motor de inteligencia artificial del sistema aideen.

**Protocolo v1.0** — `VOCAB_SIZE=64_000`, `D_GLOBAL=2048`

---

## Crates

| Crate | Estado | Descripción |
|-------|--------|-------------|
| `aideen-runtime` | ✅ Completo | GPU runtime (Metal/Vulkan/DX12/WebGPU) con wgpu |
| `aideen-backbone` | ✅ Completo | Nodo backbone — routing semántico, C+D dynamics |
| `aideen-training` | ✅ Completo | Pipeline de entrenamiento federado (4 fases) |

## Grafo de dependencias

```
aideen-backbone ──→ aideen-runtime
aideen-training ──→ aideen-runtime
aideen-runtime  ──→ (sin deps internas)
```

---

## Setup rápido

```bash
# Build de todo el workspace
cargo build-all

# Tests
cargo test-all
```

## Entrenamiento

### Fase 1 + 2 — Local (Decomposer + Backbone)

```bash
# Requiere: data/tokenizer/tokenizer.json (BPE 64K multilingual)
#           data/corpus/ (archivos .txt o .jsonl multilingüe)

cargo train-local

# O por fases separadas:
cargo run --release --bin train -- --phase decomposer
cargo run --release --bin train -- --phase backbone   # SOLO en M1 dueño
```

### Fase 3 — Federado (Expertos)

```bash
# Requiere: data/experts/{domain}/train.jsonl
#           weights/aideen_backbone_weights.safetensors (de Fase 2)

# Un dominio
cargo run --release --bin train_expert -- --domain math

# Todos los dominios
cargo train-experts

# Con Architect remoto (C+D sync)
cargo run --release --bin train_expert -- --domain all --architect 192.168.1.10:9000
```

### Fase 4 — Destilación (Expert → Backbone)

```bash
# SOLO en M1 dueño del backbone
cargo distill
```

## Dominios de expertos (18 total)

**Conocimiento:**
`math` `code` `logic` `nlp` `science` `creative`
`legal` `medical` `history` `finance` `philosophy` `multilingual`

**Razonamiento meta:**
`reasoning` `planning`

**Infraestructura cognitiva** (sin entrenamiento manual):
`memory` `synthesis` `critic` `general`

## Constantes de protocolo (NO cambiar sin versionar)

```rust
VOCAB_SIZE   = 64_000   // multilingual BPE — congelado v1.0
D_GLOBAL     = 2048     // dimensión del estado cognitivo global S_g
D_LOCAL      = 4096     // dimensión local (no sale del nodo)
MEMORY_SLOTS = 16       // slots de memoria de sesión M_t
MEMORY_DIM   = 2048     // = D_GLOBAL
```

Cambiar cualquiera de estas constantes **rompe la compatibilidad** con todos los nodos
de la red y requiere incrementar la versión mayor del protocolo (`v2.0`).

---

## Ética — invariantes no negociables

- `EthicsKernel` nunca recibe gradientes (`∂L/∂θ_ethics = 0`)
- No está en el optimizer — es un módulo separado cargado en runtime
- Se aplica a **todo** output antes de enviarlo al usuario
- No puede ser modificado por entrenamiento ni por configuración

---

## Estructura de archivos de pesos

```
weights/
├── aideen_backbone_weights.safetensors      ← Fase 2
├── aideen_decomposer_weights.safetensors    ← Fase 1
├── aideen_expert_math_weights.safetensors   ← Fase 3
├── aideen_expert_code_weights.safetensors
├── aideen_expert_logic_weights.safetensors
├── aideen_expert_nlp_weights.safetensors
├── aideen_expert_science_weights.safetensors
├── aideen_expert_creative_weights.safetensors
├── aideen_expert_legal_weights.safetensors
├── aideen_expert_medical_weights.safetensors
├── aideen_expert_history_weights.safetensors
├── aideen_expert_finance_weights.safetensors
├── aideen_expert_philosophy_weights.safetensors
├── aideen_expert_multilingual_weights.safetensors
├── aideen_expert_reasoning_weights.safetensors
└── aideen_expert_planning_weights.safetensors
```
