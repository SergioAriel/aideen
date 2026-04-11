# Roadmap — De lo actual a la visión completa

## Estado actual (branch: fixed-point-memory, 2026-04-07)

- DEQ con Picard adjoint ✓
- Slot attention per-slot (W_q, W_k, W_v, W_o por slot) ✓
- fpm_m_cache: memoria privada viva por slot, token-to-token dentro del chunk ✓
- Retain gate (low-rank r=32) con backward temporal correcto ✓
- Cross-slot read causal: lee `HistCtx[t-1]` en lugar de `MState` ✓
- Inter-chunk carry de MState ✓
- Gradiente hacia M: fluye via `loss → H_t → fpm_ctx(M_{t-1}) → retain` ✓

El flujo de gradiente es:
```
loss → lm_head → H_t → fpm_ctx (lee M_{t-1} sin stop-grad) → M_{t-1} → retain
```
No se necesita una rama separada (Memory-Augmented Output) — el path causal
ya conecta la loss con M de forma natural.

## Próximo paso — Validar que M aprende

Correr entrenamiento de 200-500 chunks y medir:
1. `retain_max` se diversifica entre slots (hoy todos en 0.953 = bias inicial)
2. `err_M` evoluciona durante el entrenamiento (no colapsa a cero ni explota)
3. `read2sig` permanece en rango razonable (0.2-1.0 es buena señal)
4. Loss mejora frente al baseline sin FPM en el corpus PG-19

Si estas métricas son positivas, la arquitectura base está validada.

## Siguiente arquitectura — Neuroplasticidad real

**Objetivo**: M modifica pesos efectivos del solve, no solo el contexto de entrada.

```
ΔW_k[t]    = low_rank(M_{k,t-1})       ← perturbación de bajo rango
W_eff_k[t] = W_base_k + ΔW_k[t]        ← pesos efectivos para este token
H_t        = solve(H ; s_t, W_eff_k[t]) ← DEQ usa pesos modificados
M_t        = update(M_{t-1}, H_t)       ← M se actualiza post-solve
```

`ΔW_k[t]` se calcula desde `M_{t-1}` — cerrado antes del Picard del token `t`.
El Jacobiano del solve no lo ve. Contractivity intacta.

Punto de entrada más conservador: hacer `W_delta` plástico por slot.
Es el peso que genera la propuesta de escritura en M — tiene sentido semántico claro.

**Cuándo**: después de validar que la memoria de contexto aprende algo útil.

## Clawbot — Memoria persistente por usuario

**Objetivo**: M_user persiste entre sesiones por usuario.

Stack:
```
W_base  (AIDEEN preentrenado, igual para todos)
W_bot   (fine-tuning: identidad, rol, herramientas)
M_user  (por usuario, persiste entre sesiones, nunca se sincroniza)
```

Cambios de infraestructura (no arquitectura):
- guardar M_user al cerrar sesión
- cargar M_user al abrir sesión
- inicializar en ceros para usuarios nuevos

**Cuándo**: la infraestructura es simple y puede hacerse en paralelo con la validación.
No requiere neuroplasticidad completa para ser útil.

## Escalado

- Flash Attention para ctx_len > 512
- d_r = 1024+ (extrapolación scaling law: loss ≈ 2.38)
- Entrenamiento distribuido entre nodos RDNA 3

**Cuándo**: después de validar la arquitectura completa a escala pequeña.

## Principios

1. Validar antes de extender
2. No romper la contractivity del Picard — es el invariante central
3. W_base nunca cambia en runtime — M cambia token a token
4. El benchmark PG-19 es la métrica de verdad — no runs cortos de 20 chunks
