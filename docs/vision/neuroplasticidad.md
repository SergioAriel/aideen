# Neuroplasticidad en AIDEEN

## La distinción fundamental

Hay dos formas en que un modelo puede "recordar":

**Memoria de contexto**: el modelo recibe información adicional como input y la usa para condicionar su output. Los pesos no cambian. Es lo que hace un transformer con KV-cache, o AIDEEN hoy con `fpm_ctx`.

**Neuroplasticidad**: el modelo modifica sus propios pesos efectivos durante el procesamiento. El sistema que computa cambia con la experiencia, no solo su input.

AIDEEN actualmente tiene memoria de contexto. La visión es neuroplasticidad.

## Por qué importa la diferencia

Con memoria de contexto:
- el modelo *recuerda* lo que pasó
- pero procesa siempre con la misma función

Con neuroplasticidad:
- el modelo *aprende* durante la inferencia
- la función de procesamiento se adapta al contexto
- es análogo a la plasticidad sináptica biológica

## El mecanismo neuroplástico objetivo

```
ΔW_k[t] = low_rank(M_{k,t-1})     ← perturbación de bajo rango generada desde M
W_eff_k[t] = W_base_k + ΔW_k[t]   ← pesos efectivos para el token t

H_t = solve(H ; s_t, W_eff[t])     ← DEQ usa pesos modificados
M_t = update(M_{t-1}, H_t)         ← M se actualiza con el resultado
```

**Por qué es seguro dentro del DEQ:**
`ΔW_k[t]` se calcula desde `M_{t-1}` — cerrado antes de que empiece el Picard del token `t`. El Jacobiano del solve ve `W_eff[t]` como constante. Contractivity intacta.

## Qué pesos modifica la plasticidad

No todos los pesos necesitan ser plásticos. Los candidatos naturales por slot:

- `W_delta` — el peso que genera la propuesta de escritura en M. Tiene más sentido que sea plástico: el slot aprende *cómo escribir* en su memoria
- `W_v` — la proyección de valor en la attention. Modifica *qué información extrae* el slot del contexto
- `W_q` / `W_k` — más arriesgado, afecta la estructura de atención directamente

El punto de partida más conservador y más fiel a la idea original es hacer `W_delta` plástico por slot.

## Dónde NO aplica neuroplasticidad

- `W_base` del modelo nunca cambia en runtime — solo con reentrenamiento
- `lm_head`, embeddings — función pura de la tarea, no del contexto
- Los pesos de slots que manejan estructura sintáctica básica — no necesitan adaptarse

## Separación de timescales

| Timescale | Qué cambia | Cuándo |
|---|---|---|
| Dentro del token (Picard) | Nada — todo es constante | Cada iteración |
| Entre tokens | M evoluciona, W_eff se recalcula | Cada token |
| Entre chunks | MState persiste | Cada chunk |
| Entre sesiones | M_user persiste | Cada sesión |
| Reentrenamiento | W_base se actualiza | Releases del modelo |

La neuroplasticidad opera en los timescales de token, chunk y sesión. W_base es inmutable en runtime.

## Estado actual

La implementación actual tiene:
- `fpm_m_cache`: memoria privada viva por slot dentro del chunk ✓
- `retain gate`: gate aprendible que controla la persistencia de M ✓
- `MState`: persistencia entre chunks ✓
- Plasticidad de pesos efectivos: **pendiente**
- Stop-grad en M: presente, se puede eliminar cuando la lectura sea causal ✓ (roadmap)
