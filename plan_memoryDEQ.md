# Plan Final de Implementación: Memoria Temporal Selectiva y DEQ-Compatible para AIDEEN

## Resumen

Se implementará una memoria temporal fuerte, inspirada en Mamba, pero integrada de forma compatible con el fixed-point del DEQ.

La estrategia productiva será:

1. Mantener la memoria temporal **fuera del loop Picard**.
2. Hacer que la historia entre al DEQ como **contexto fijo por token**, per-slot, proyectado, con magnitud controlada y gate explícita.
3. Entrenar primero la **interfaz histórica**.
4. Entrenar después la **dinámica temporal externa**:
   - primero simple,
   - luego selectiva input-dependent.
5. Mantener `fixed_mamba` e `init_mamba` solo como ablations.
6. No aumentar el scratch stride a `7K+1`; se mantendrá `6K+1` con reuso por lifetime.

Este plan es la ruta productiva. La integración interna de memoria dentro del DEQ queda explícitamente postergada a una fase experimental posterior.

---

## 1. Failure mode objetivo

El objetivo es corregir esta falla estructural:

- `no_mamba` converge limpio.
- cualquier intento actual de reintroducir memoria temporal (`fixed_mamba`, `init_mamba`) degrada:
  - contractividad,
  - convergencia,
  - o costo total.

### Causa estructural
La historia temporal entra al DEQ con una interfaz incorrecta:
- demasiado cruda,
- sin proyección adecuada,
- sin control correcto de magnitud,
- sin suficiente especialización por slot.

El problema no es “tener memoria”, sino **cómo la memoria modula el operador del DEQ**.

---

## 2. Arquitectura objetivo

## 2.1 Variables

Para token `t` y slot `k`:

- `s_t ∈ R^D`: input actual
- `M_{t-1,k} ∈ R^D`: memoria temporal externa previa
- `h_{t,k}^{(\ell)} ∈ R^D`: iterado Picard
- `h_{t,k}^* ∈ R^D`: estado convergido DEQ
- `c_{t,k} ∈ R^D`: contexto histórico fijo del token
- `x_{t,k} ∈ R^D`: proyección temporal de `h_t^*`

---

## 2.2 Interfaz histórica per-slot

### Proyección histórica

\[
u_{t,k} = W_{hist}^{shared} M_{t-1,k} + d_k \odot M_{t-1,k}
\]

donde:

- `W_hist^{shared} ∈ R^{D×D}`
- `d_k ∈ R^D`, uno por slot

### Inicialización

- `W_hist^{shared} = I + 0.01\xi`
- `d_k = 0`

### Justificación
- evita bypass identidad vía `d_k`
- mantiene el canal histórico vivo desde el inicio
- no fuerza a la memoria a entrar “cruda”

---

## 2.3 Control de magnitud del canal histórico

Definir:

\[
r_u = RMS(u_{t,k}), \qquad \tau_t = RMS(inj_t)
\]

\[
\tilde u_{t,k} = \frac{u_{t,k}}{\max\left(1,\frac{r_u}{\tau_t + \epsilon}\right)}
\]

### Decisión importante
`\tau_t` se tratará como **detach** en el backward del cap.

### Justificación
- `\tau_t` es una escala de referencia del input, no un camino de entrenamiento hacia `W_in`
- evitar gradientes espurios `\partial \tau_t / \partial W_in`
- mantener el canal histórico desacoplado del aprendizaje de amplitud del input principal

Esto debe implementarse como invariante explícita.

---

## 2.4 Gate con piso positivo

\[
\alpha_k = \alpha_{min} + (\alpha_{max} - \alpha_{min}) \sigma(g_k)
\]

con:

- `α_min = 0.05`
- `α_max = 0.25`

Inicialización:
- `α_k(0) = 0.10`

### Justificación
- evita canal histórico muerto desde el arranque
- mantiene control estructural de escala

### Contexto final

\[
c_{t,k} = \alpha_k \tilde u_{t,k}
\]

---

## 2.5 Operador DEQ

Para cada iteración Picard:

\[
q_k = W_q h_k^{(\ell)},\quad
k_j = W_k h_j^{(\ell)},\quad
v_j = W_v h_j^{(\ell)}
\]

\[
a_{k,j} = softmax_j\left(\frac{q_k^\top k_j}{\sqrt{D}}\right)
\]

\[
attn_k(h^{(\ell)}) = W_o \sum_j a_{k,j} v_j
\]

\[
inj_t = W_{in}s_t
\]

\[
z_k^{(\ell)} = attn_k(h^{(\ell)}) + inj_t + c_{t,k}
\]

\[
f_k(h^{(\ell)}; s_t, M_{t-1}) = RMSNorm(z_k^{(\ell)})
\]

\[
h_k^{(\ell+1)} = \beta f_k(h^{(\ell)}; s_t, M_{t-1}) + (1-\beta)h_k^{(\ell)}
\]

### Propiedad clave
- `c_{t,k}` es fijo durante todas las iteraciones del token
- `M_{t-1,k}` no cambia dentro del DEQ
- el DEQ sigue resolviendo un operador fijo en `h`

---

## 3. Memoria temporal externa

## 3.1 Fase temporal simple

\[
x_{t,k} = W_x h_{t,k}^*
\]

\[
a = \sigma(-A_{log})
\]

\[
\tilde m_{t,k} = a \odot M_{t-1,k} + (1-a)\odot x_{t,k}
\]

\[
M_{t,k} = W_{out}\tilde m_{t,k}
\]

Esta fase sirve para estabilizar la interfaz histórica y el backward temporal básico.

## 3.2 Fase temporal selectiva

Luego se reemplaza por:

\[
\Delta_{t,k} = softplus(W_\Delta h_{t,k}^* + b_\Delta)
\]

\[
a_{t,k} = \exp(-\Delta_{t,k} \odot A)
\]

\[
x_{t,k} = W_x h_{t,k}^*
\]

\[
\tilde m_{t,k} = a_{t,k} \odot M_{t-1,k} + (1-a_{t,k})\odot x_{t,k}
\]

\[
M_{t,k} = W_{out}\tilde m_{t,k}
\]

### Justificación
Esto devuelve la selectividad input-dependent que la memoria actual no tiene, sin meter dinámica temporal dentro de Picard.

---

## 4. Backward de la interfaz histórica

## 4.1 Gradiente hacia el contexto

Como `c_{t,k}` entra sumado en `z_k`:

\[
g^c_{t,k} = \frac{\partial L}{\partial c_{t,k}} = g^{comb}_{t,k}
\]

## 4.2 Backward de la gate

\[
\frac{\partial L}{\partial g_k}
=
(\alpha_{max}-\alpha_{min})\sigma(g_k)(1-\sigma(g_k))
\sum_t \left\langle g^c_{t,k}, \tilde u_{t,k} \right\rangle
\]

## 4.3 Backward exacto del cap

Sea:

\[
\tilde u = s u,\qquad
s = \min\left(1,\frac{\tau}{r}\right),\qquad
r = \sqrt{\frac{1}{D}\sum_i u_i^2 + \epsilon}
\]

y \(g = \partial L / \partial \tilde u\).

### Rama no clipeada
\[
\frac{\partial L}{\partial u} = g
\]

### Rama clipeada
\[
\frac{\partial L}{\partial u}
=
s g
-
\frac{s}{D r^2} u (u^\top g)
\]

### Requisito
Esta fórmula se implementará exacta.  
No se acepta la aproximación incompleta `s g`.

## 4.4 Backward de la proyección

\[
\frac{\partial L}{\partial W_{hist}^{shared}}
+=
\sum_{t,k} g^u_{t,k}\otimes M_{t-1,k}
\]

\[
\frac{\partial L}{\partial d_k}
+=
\sum_t g^u_{t,k}\odot M_{t-1,k}
\]

\[
g^{(deq)}_{M_{t-1,k}}
=
(W_{hist}^{shared})^\top g^u_{t,k}
+
d_k \odot g^u_{t,k}
\]

---

## 5. Backward temporal externo

## 5.1 Fase simple

### Forward temporal
\[
M_t = g(M_{t-1}, h_t^*)
\]

### Backward temporal
Se hará por `TBPTT` con chunks:

- primero `L = 16`
- luego `L = 32`

### Orden
1. LM backward produce `dl_dh_t`
2. backward temporal produce:
   - `g_h^{temporal}(t)`
   - gradientes de `W_x`, `W_out`, `A_log`
3. se forma:
\[
b_t = dl_dh_t + g_h^{temporal}(t)
\]
4. staged Picard resuelve el DEQ para ese `b_t`

## 5.2 Fase selectiva

Agregar:

\[
\frac{\partial L}{\partial h_t^*}
\leftarrow
\frac{\partial L}{\partial h_t^*}
+
W_\Delta^\top \frac{\partial L}{\partial \Delta_t}
\]

\[
\frac{\partial L}{\partial \Delta_t}
=
\frac{\partial L}{\partial a_t}\odot (-a_t \odot A)
\]

Esto se implementará explícitamente en el shader temporal.

---

## 6. Scratch y memoria

## 6.1 Restricción

Se mantiene:

\[
scratch\_stride = D(6K + 1) + K^2
\]

No se aumenta a `7K+1`.

## 6.2 Reuso por lifetime

- `q_base`: Q
- `k_base`: K
- `v_base`: V
- `attn_base`: salida de atención / temporario
- `mamba_base`:
  - durante DEQ: `c_{t,k}`
  - después: `M_t`
- `signal_base`: `inj_t`
- `m_inner_base`:
  - durante DEQ: temporario
  - después: `\tilde m_t`
- `attn_weight_base`: pesos de atención

### Justificación
Se evita el impuesto de bandwidth y cache de un stride mayor.

---

## 7. Cambios por archivo

## 7.1 `mamba_slot_reasoning.rs`

Agregar:
- `w_hist_shared: DMatrix<f32>`
- `hist_slot_scale: DMatrix<f32>` (`K×D`)
- `hist_gate_logit: DVector<f32>` (`K`)
- `w_delta: DMatrix<f32>` (`D×D`) para fase selectiva
- `b_delta: DVector<f32>` (`D`) para fase selectiva

### Inicialización
- `w_hist_shared = I + 0.01\xi`
- `hist_slot_scale = 0`
- `hist_gate_logit` para gate efectiva `0.10`
- `w_delta`, `b_delta` inicializados pero congelados hasta fase selectiva

### Renorm
- `w_hist_shared` **no** usa el threshold `0.10`
- razón:
  - `W_hist` no entra al Jacobiano respecto a `h`
  - no afecta contractividad del DEQ
  - la escala efectiva ya queda controlada por el cap relativo a `\tau_t`

Si se aplica control numérico, será solo un clip suave opcional (`σ ≤ 1.5`), no una amputación agresiva.

## 7.2 `gpu_deq.rs`

Agregar buffers:
- `hist_w_buf`
- `hist_slot_scale_buf`
- `hist_gate_buf`
- `w_delta_buf`
- `b_delta_buf`

Actualizar bind groups de:
- forward DEQ
- fused DEQ update
- temporal backward

## 7.3 `deq_forward.wgsl`

Nuevo modo:
- `AIDEEN_DEQ_HIST_GATED=1`

Implementar:
1. lectura de `M_{t-1,k}`
2. cálculo de `u`
3. cálculo de `\tau_t`
4. cap relativo
5. gate
6. escritura de `c_{t,k}` en `mamba_base`
7. `combined = attn + inj + c`

Mantener update temporal post-convergencia.

## 7.4 `fused_deq_update.wgsl`

Agregar update de:
- `W_hist_shared`
- `hist_slot_scale`
- `hist_gate_logit`

Debe incluir el backward exacto del cap.

## 7.5 Nuevo shader: `temporal_mamba_backward.wgsl`

Fase simple:
- gradientes de `W_x`, `W_out`, `A_log`

Fase selectiva:
- gradientes de `W_\Delta`, `b_\Delta`

## 7.6 `trainer.rs`

Agregar stage:
- `AttnHistMamba`

Orden de ejecución:
1. LM backward
2. si fase 1:
   - staged Picard con `dl_dh`
3. si fase temporal:
   - temporal backward externo
   - formar `b_total`
   - staged Picard con `b_total`
4. fused update DEQ
5. update temporal

---

## 8. Test cases y aceptación

## 8.1 Tests numéricos

1. `hist_gated` small reference:
- `D=16`, `K=2`
- coseno `> 0.99`

2. Grad-check del cap:
- rama clipeada
- coseno `> 0.99`

3. Grad-check de:
- `W_hist_shared`
- `hist_slot_scale`
- `hist_gate_logit`

4. Grad-check temporal:
- fase simple: `W_x/W_out/A_log`
- fase selectiva: `W_\Delta/b_\Delta`

## 8.2 Stress tests

Baselines:
- `AIDEEN_DEQ_NO_MAMBA=1`
- `AIDEEN_DEQ_FIXED_MAMBA=1`
- `AIDEEN_DEQ_INIT_MAMBA=1`

Candidate:
- `AIDEEN_DEQ_HIST_GATED=1`

Runs:
- `20` iter
- `100` iter

## 8.3 Acceptance criteria

### Fase 1
- `mode = NORMAL`
- `conv = OK`
- `hit = 0`
- `contr < 0.30`
- `loss20 <= loss20(no_mamba) + 1%`
- `loss100 < loss100(no_mamba)`

### Canal histórico
- `hist_ctx_rms / inj_rms` mediano en `[0.06, 0.25]`
- máximo `< 0.35`
- gate efectiva media por slot en `[0.08, 0.18]`

### Criterio de canal muerto
Si tras `100` steps:
\[
median(hist\_ctx\_rms / inj\_rms) < 0.06
\]
la fase falla.

### Fase temporal
- gradientes no nulos y finitos
- sin `BOOST/FAIL`
- costo por iteración `≤ 1.5×` baseline `no_mamba`

---

## 9. Assumptions y defaults

1. `\tau_t` se trata como `detach` en el backward del cap.
2. `d_k` inicia en `0`.
3. `W_hist_shared` inicia cerca de identidad.
4. `W_hist_shared` no se renormaliza a `0.10`.
5. La gate tiene piso positivo.
6. El canal histórico entra per-slot.
7. La memoria temporal selectiva es el objetivo final.
8. El backward temporal se hace por `TBPTT`.
9. `fixed_mamba` e `init_mamba` quedan como ablations.
10. No se aumenta el scratch stride.

---

## 10. Resultado esperado

Al finalizar la fase selectiva, AIDEEN debe tener:

- un canal histórico vivo desde el inicio,
- una memoria temporal realmente selectiva por input,
- una interfaz histórica que preserve tanto dirección como magnitud útil,
- convergencia limpia del DEQ,
- y un costo controlado mediante backward temporal externo truncado.

Ese será el baseline serio antes de considerar una memoria realmente interna al DEQ.
