# Arquitectura de AIDEEN

## Qué es

AIDEEN es un modelo de lenguaje con dos timescales de dinámica acopladas:

- **Timescale rápida**: el estado de activación `H` se resuelve a punto fijo dentro de cada token via iteración de Picard (DEQ)
- **Timescale lenta**: la memoria `M` evoluciona causalmente token a token, persiste entre chunks y entre sesiones

La separación de timescales es deliberada. Permite que el solve de `H` sea estable y convergente, mientras `M` acumula información a lo largo del tiempo sin interferir con la contractivity del Picard.

## Las tres familias que lo componen

| Componente | Descripción | Referencia |
|---|---|---|
| DEQ — Deep Equilibrium | H se resuelve como punto fijo: `h* = f(h*; s_t, M_{t-1})` | Bai et al. 2019 |
| Slot Attention | Múltiples slots compiten y se coordinan sobre el input | Locatello et al. 2020 |
| Fast Weights / Two-Timescale | M modifica el comportamiento del modelo token a token | Ba et al. 2016, Titans 2024 |

La combinación específica de los tres no existe publicada. El aporte original es el acoplamiento two-timescale dentro de un DEQ con estructura de slots.

## Caracterización formal

> **Two-timescale recurrent DEQ con slot-structured memory**

El flujo semántica objetivo (no necesariamente el estado actual de implementación):

```
token t:
  read_t  = Read(M_{t-1})           ← lectura de memoria causal
  H_t     = solve(H ; s_t, read_t)  ← DEQ estable, M_{t-1} es constante durante el solve
  M_t     = update(M_{t-1}, H_t)    ← actualización plástica post-solve
  output_t = Head(H_t, M_t)         ← output enriquecido con memoria
```

`M_{t-1}` está cerrado antes de que empiece el Picard del token `t`. Por eso el Jacobiano del solve no incluye `∂M/∂h` — la contractivity se preserva.

## Estructura de slots

Cada slot `k` mantiene:
- su estado de activación `h_k[t]` — resuelto por el DEQ
- su memoria privada `m_k[t]` — evoluciona token a token

La coordinación entre slots ocurre en dos niveles:
1. **Dentro del DEQ**: los slots se atienden mutuamente via Q/K/V sobre `H_curr`
2. **En la fase de output**: los slots consultan las memorias actualizadas de otros slots (objetivo futuro)

## Relación con la literatura

**Fast Weights (Ba et al. 2016)**: propone memoria que modifica pesos efectivos durante inferencia. AIDEEN actualmente modifica el contexto del solve, no los pesos. Neuroplasticidad real (modificar `W_eff`) es una extensión natural y está en el roadmap.

**Titans (Google, 2024)**: memoria de largo plazo implementada como red pequeña cuyos pesos se actualizan por gradient descent online. Es el referente más cercano a la dirección neuroplástica de AIDEEN.

**Fixed-Point Memory / SSM**: el retain gate de AIDEEN es análogo a un GRU gate sobre memoria persistente. A_log, W_forget son heredados de esta familia.

## Estado actual vs. visión objetivo

| Aspecto | Estado actual | Objetivo |
|---|---|---|
| Lectura de M en el solve | `MState` (snapshot por chunk) | `HistCtx[t-1]` (causal token→token) |
| Stop-grad en M | Explícito (protege Picard) | Eliminable cuando M sea causal |
| Tipo de plasticidad | Contexto (vector sumado) | Pesos efectivos `W_eff = W_base + ΔW(M)` |
| Gradiente hacia M | Truncado (semilla cero) | Causal real desde la loss |
| Cross-slot read | Sobre `MState` congelado | Sobre `HistCtx[t-1]` actualizado |
