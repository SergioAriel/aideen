---
name: provisional-results
description: "Documentar resultados buenos como baseline provisional (no default) con evidencias, configuración exacta, y criterios de invalidez. Usar cuando aparezcan runs estables/óptimos en debugging, tuning o cambios arquitectónicos y se necesite registrar una guía provisional hasta que sea rebatida."
---

# Provisional Results

Objetivo: registrar resultados buenos como **baseline provisional**, sin convertirlos en default. La guía debe ser reproducible, falsable y fácil de invalidar si aparece nueva evidencia.

## Regla central

- **Nunca** promover a default por “un run bueno”.
- **Siempre** registrar condiciones exactas, métricas y umbrales de invalidez.

## Qué documentar (siempre)

1. **Contexto**
   - qué se estaba probando (ej: “V dinámica en DEQ con hist_gated”)
   - objetivo del run (estabilidad, loss, convergencia, etc.)

2. **Configuración exacta**
   - comando completo (con envs)
   - seed, iters, dataset, perfil
   - hash/estado de archivos críticos si cambió código

3. **Resultados**
   - métricas clave: `mode`, `conv`, `contr`, `maxΔ`, `loss`, `attn_ent`, `inj_rms`, y las que sean necesarias tener en cuenta.
   - “antes vs después” si aplica

4. **Criterios de invalidez**
   - qué resultado futuro lo refuta (ej: “si seed 13 muestra DEQ‑INVALID en step 3”)
   - qué cambio de código lo vuelve obsoleto

5. **Alcance**
   - “válido solo para estos seeds / este dataset / este perfil”

## Plantilla (copiar y completar)

```
### [YYYY-MM-DD] Baseline provisional: <titulo corto>
**Contexto**: <qué se estaba probando>
**Objetivo**: <qué debía mejorar>

**Configuración exacta**
- Cmd: `<command line>`
- Env: <lista completa de envs>
- Seed: <n>
- Iters: <n>
- Dataset: <nombre>
- Perfil: <stable/emerg/etc>
- Código: <hash o lista de archivos tocados>

**Resultados**
- mode/conv: <NORMAL/OK o EMERG/FAIL>
- contr/maxΔ: <valores>
- loss: <rango>
- métricas extra: <q/k/v, attn_ent, inj_rms...>

**Criterios de invalidez**
- <si ocurre X en seed Y o en configuración Z, se invalida>
- <si cambia archivo A/B/C, se debe revalidar>

**Alcance**
- <hasta qué condiciones aplica>
```

## Cómo usar esta skill

1. Cada vez que aparezca un “run bueno”, registra una entrada provisional.
2. **No** cambies defaults con esa entrada.
3. Si aparece evidencia contradictoria, marca la entrada como **invalidada** y explica por qué.
4. Solo promover a default si:
   - se valida en múltiples seeds
   - el comportamiento se mantiene en runs largos
   - no hay contradicciones posteriores

## Interacción con root-cause-first

Si el resultado bueno aparece después de un fix:
- agrega **qué invariante se restauró**
- agrega **por qué no es un parche**

