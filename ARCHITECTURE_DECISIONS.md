# Aideen AI — Architecture Decisions

## ADR-001: GenerationStrategy — Cómo generar texto desde H*

**Estado:** EN EVALUACIÓN — ninguna estrategia seleccionada definitivamente.
**Contexto:** H* es el punto fijo del DEQ (8 slots × D_R). Se necesita decidir cómo
convertirlo en una secuencia de tokens. Hay 3 alternativas viables.

---

### Estrategia A: SlotDirect

```
H* (8 slots × D_R)
   ↓
Cada slot → LmHead → 1 token
   ↓
Secuencia de hasta K=8 tokens
```

- ✅ Más simple, sin capas extra
- ✅ K slots ya tienen estado del DEQ — no hay redundancia
- ✅ Latencia mínima (una multiplicación de matriz)
- ❌ Máximo K tokens por respuesta
- ❌ Los slots son "razonadores paralelos", no posiciones secuenciales

**Cuándo gana:** Si el DEQ aprende a especializar slots por posición de token.

---

### Estrategia B: Decoder (actual)

```
H* ──FiLM──► scale/bias por capa
<bos> → Fixed-Point Memory layer 0..N → LmHead → tokens
```

- ✅ Sequencias de largo arbitrario
- ✅ H* condiciona cada capa via FiLM (guía semántica)
- ✅ El decoder tiene su propia memoria del texto generado
- ❌ Capas extra = más parámetros, más latencia
- ❌ Posible redundancia con el DEQ

**Cuándo gana:** Si la query requiere respuestas largas y coherentes.

---

### Estrategia C: DeqAutoReg

```
query_0 → DEQ → H*_0 → LmHead → token_0
query_1 = query_0 + token_0 → DEQ → H*_1 → LmHead → token_1
query_2 = query_1 + token_1 → DEQ → H*_2 → LmHead → token_2
...
```

- ✅ Autoregressivo a nivel DEQ — cada token es el resultado de razonamiento completo
- ✅ Sin decoder extra — reutiliza el DEQ ya construido
- ✅ Coherencia máxima: cada token "entiende" todos los anteriores
- ❌ Costo = max_iters × tokens_a_generar (muy costoso sin GPU)
- ❌ El DEQ necesita converger para cada token

**Cuándo gana:** Si el DEQ es rápido en GPU y las respuestas requieren coherencia profunda.

---

### Plan de evaluación

Todas las estrategias están implementadas en `aideen-backbone/src/generation_strategy.rs`.
El benchmark `cargo test -p aideen-backbone benchmark_all_three_strategies -- --nocapture` compara:

1. Latencia por token
2. Diversidad de tokens generados (no degeneración repetitiva)
3. Sensibilidad al input (tokens distintos para queries distintas)

**La estrategia ganadora se decide con pesos entrenados, no con pesos random.**

---

### Resultados con pesos random, D_R=1024 (histórico)

```
SlotDirect   diversity=1.00  elapsed=24ms
Decoder      diversity=0.38  elapsed=1905ms
DeqAutoReg   diversity=0.12  elapsed=31903ms
```

### Resultados con pesos random, D_R=256 + Picard β=0.9

```
SlotDirect   diversity=1.00  elapsed=6ms       ← 4x más rápido
Decoder      diversity=0.12  elapsed=133ms     ← 14x más rápido
DeqAutoReg   diversity=0.12  elapsed=3662ms    ← 8.7x más rápido
Ganador por diversidad: SlotDirect
```

**Nota:** Decoder y DeqAutoReg ahora colapsan igual (diversity=0.12).
Esto se debe a la Picard β-relación: β=0.9 suaviza H* y reduce diversidad
trivial de tokens con pesos random. Con pesos entrenados el DEQ generará
H* genuinamente distintos por query.

**Hipótesis a validar con entrenamiento:**
- SlotDirect: ¿Los slots aprenden posiciones semánticas distintas?
- Decoder: ¿El FiLM condicionado por H* reduce la degeneración?
- DeqAutoReg: ¿Cada token genera un H* genuinamente diferente con pesos entrenados?

---

## ADR-002: D_R dimension

**Estado:** PENDIENTE

- Mobile: D_R = 256
- Desktop: D_R = 512
- Cloud/training: D_R = 1024 (actual)

Cambiar en `aideen-core/src/state.rs`.

---

## ADR-003: Spectral Normalization en FixedPointMemoryReasoning

**Estado:** PENDIENTE

Implementar para garantizar convergencia DEQ con pesos entrenados.
Sin spectral norm, el DEQ puede divergir o oscilar con pesos no-triviales.
