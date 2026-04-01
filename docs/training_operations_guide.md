# Training Operations Guide

Guía operativa para correr training, reporting e inferencia en AIDEEN sin caer en
los problemas que ya vimos en esta tanda: comandos mal formados, falta de visibilidad,
TPS engañoso y corridas que parecen “colgadas” cuando en realidad están tokenizando,
compilando o esperando un primer hito de progreso.

## Objetivo

Este documento no intenta congelar el sistema para siempre. AIDEEN sigue en fase de
análisis y mejora. Pero sí fija una base operativa estable para:

1. lanzar training real con una configuración conocida;
2. obtener métricas visibles y comparables;
3. correr inferencia sobre checkpoints sin depender del chat interactivo;
4. reducir errores operativos que no deberían volver a ocurrir.

---

## Estado actual del sistema

### Qué ya está estabilizado

- El `train` real corre bien con el path fused actual.
- El benchmark/runner ya distingue perfiles de throughput vs reporting.
- El progreso ahora diferencia:
  - `tps_win`: throughput de ventana
  - `tps_run`: throughput acumulado del run
- Cuando `AIDEEN_PROGRESS_EVERY>0`, esos TPS ya miden trabajo GPU completado.
  El trainer sincroniza en cada corte de progreso para evitar números inflados por comandos en cola.
- Si no hay una loss confiable visible, el trainer muestra `loss=n/a` en vez de `0.0000`.
- La inferencia rápida ya tiene un bin dedicado:
  - `aideen-training-lab/src/bin/infer.rs`
- La inspección de checkpoints ya tiene un script dedicado:
  - `report_checkpoint.sh`

### Qué sigue siendo deuda o limitación

- Algunos defaults siguen siendo demasiado sensibles al perfil de uso.
- El launch en background vía `nohup &` fue poco confiable en este entorno; la sesión TTY
  persistente funcionó mejor.
- El checkpoint actual `model_large` todavía no genera texto útil.
- La rama histórica existe, pero el checkpoint observado mostró `hist_scale` casi apagada.

---

## Regla principal

No usar un solo comando para todo.

En AIDEEN hoy hay tres tareas diferentes:

1. **training real de throughput**
2. **training con reporting/calidad visible**
3. **inferencia / evaluación de checkpoint**

Cada una requiere un perfil distinto.

---

## 1. Training real estable

Este es el comando base recomendado para un run real sobre el corpus grande.

### Comando

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=model_large \
  AIDEEN_BATCH_SIZE=8 \
  AIDEEN_CTX_LEN=512 \
  AIDEEN_LM_FUSED_B19=1 \
  AIDEEN_DEQ_HIST_GATED=1 \
  AIDEEN_LOSS_READBACK_EVERY=0 \
  AIDEEN_TPS_SYNC_EVERY=0 \
  AIDEEN_VAL_EVERY=0 \
  AIDEEN_PROGRESS_EVERY=20 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/aideen/corpus_combined.txt \
    --epochs 12 \
    --log-every 1 \
    --save-every 0
```

### Qué hace cada flag importante

- `AIDEEN_BATCH_SIZE=8`
  - perfil de throughput alto razonable para M1 Pro
- `AIDEEN_CTX_LEN=512`
  - mejor ocupación que `256` en este régimen
- `AIDEEN_LM_FUSED_B19=1`
  - usar path fused rápido del LM
- `AIDEEN_DEQ_HIST_GATED=1`
  - mantener history activa en el sistema real
- `AIDEEN_LOSS_READBACK_EVERY=0`
  - no bloquear training por loss readbacks intra-step
- `AIDEEN_TPS_SYNC_EVERY=0`
  - no meter syncs de observabilidad en el hot path
- `AIDEEN_VAL_EVERY=0`
  - no validar durante training throughput
- `AIDEEN_PROGRESS_EVERY=20`
  - da visibilidad sin demasiado ruido
- `AIDEEN_CHECKPOINT_BASE=model_large`
  - base del checkpoint que se usaría si activás guardado
  - útil para corridas comparativas sin pisar un checkpoint previo

### Cuándo usarlo

- runs largos
- throughput real
- entrenamiento de checkpoint principal

### Qué no esperar

- loss útil en cada chunk
- `val_loss` intermedia
- diagnóstico fino de calidad

Este perfil está optimizado para **correr**, no para explicar por qué aprende o no.

---

## Corpus recomendado para pruebas serias

No usar `corpus_combined.txt` directamente como corpus principal de pretraining.

La auditoría mostró que ese archivo quedó dominado por bloques:

- `USER:`
- `ASSISTANT:`
- `SYSTEM:`

Eso sirve para una etapa de instruct/chat, pero no para el pretraining base del
modelo.

### Generar corpora limpios y separados

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
python3 prepare_training_corpora.py
```

Esto genera en `/Users/sergiosolis/Programacion/aideen`:

- `corpus_pretrain_minimal.txt`
- `corpus_chat_instruct.txt`

### Qué contiene cada uno

- `corpus_pretrain_minimal.txt`
  - Rust Book
  - FineWeb limpio extraído del corpus combinado
  - documentación local del proyecto:
    - `README.md`
    - `ARCHITECTURE.md`
    - `PLAN.md`
    - `ARCHITECTURE_DECISIONS.md`
    - `docs/distributed_training_users.md`

- `corpus_chat_instruct.txt`
  - el bloque conversacional `USER:/ASSISTANT:/SYSTEM:` separado para una etapa
    futura de finetune/chat

### Smoke test recomendado

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
AIDEEN_LADDER_BASE=model_clean_probe \
AIDEEN_CORPUS_FILE=/Users/sergiosolis/Programacion/aideen/corpus_pretrain_minimal.txt \
AIDEEN_TINY_EPOCHS=1 \
AIDEEN_CORPUS_EPOCHS=1 \
AIDEEN_MAX_CHUNKS=40 \
AIDEEN_MAX_CHUNKS_CORPUS=40 \
./train_learning_ladder.sh both
```

Este run no busca calidad final. Busca validar que:

1. el modelo aprenda una distribución más coherente;
2. no reaparezcan tokens degenerados tipo `ASS/USER/IST/ANT`;
3. la historia siga viva sin que el corpus la tape.

---

## 2. Training con reporting visible

Cuando el objetivo es evaluar calidad/aprendizaje y no solo throughput, usar el perfil
de reporting.

### Comando recomendado

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=model_report \
  AIDEEN_BATCH_SIZE=4 \
  AIDEEN_CTX_LEN=512 \
  AIDEEN_LM_FUSED_B19=1 \
  AIDEEN_DEQ_HIST_GATED=1 \
  AIDEEN_LOSS_READBACK_EVERY=20 \
  AIDEEN_TPS_SYNC_EVERY=20 \
  AIDEEN_VAL_EVERY=200 \
  AIDEEN_PROGRESS_EVERY=20 \
  AIDEEN_MAX_CHUNKS=200 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
    --epochs 1 \
    --log-every 1 \
    --save-every 0
```

### Cuándo usarlo

- medir `loss` visible
- comparar `history on/off`
- validar que una modificación realmente mejora entrenamiento
- runs cortos y comparables

### Qué esperar

- progreso con `loss=...` o `loss=n/a` honesto
- `tps_win`
- `tps_run`
- `tps_epoch` al final
- checkpoint separado si activás `--save-every > 0`

---

## 3. Inferencia rápida con checkpoint

Usar el bin `infer`.

### Comando básico

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
cargo run --release --features wgpu -p aideen-training --bin infer -- \
  --model model_large \
  --prompt "The Rust Programming Language is" \
  --max-tokens 48 \
  --temperature 0.15 \
  --top-p 0.75 \
  --top-k 6 \
  --rep-penalty 1.15
```

### Con stats del checkpoint

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
cargo run --release --features wgpu -p aideen-training --bin infer -- \
  --model model_large \
  --stats \
  --prompt "The Rust Programming Language is" \
  --max-tokens 48 \
  --temperature 0.15 \
  --top-p 0.75 \
  --top-k 6 \
  --rep-penalty 1.15
```

### Qué informa

- tiempo de carga
- tiempo de generación
- `tok/s`
- stats de:
  - embeddings
  - LM head
  - history params

---

## 4. Reporte reproducible de checkpoint

Usar el script:

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./report_checkpoint.sh model_large
```

### Qué hace

- corre inferencia con `history on`
- corre inferencia con `history off`
- usa prompts fijos
- imprime stats del checkpoint

### Para qué sirve

- comparar ramas de entrenamiento
- inspeccionar si history está viva
- ver si el modelo sigue en “token soup”

---

## 5. Causas típicas de “parece colgado”

Estas fueron causas reales observadas.

### 1. Comando mal formado

El bin `train` **no** usa subcomando `train`.

### Correcto

```bash
cargo run --release --features wgpu -p aideen-training --bin train -- --file ...
```

### Incorrecto

```bash
cargo run --release --features wgpu -p aideen-training --bin train -- train --file ...
```

Ese `train` extra puede hacer que el proceso muera o que el launch falle antes de arrancar bien.

---

### 2. Sin progreso visible

Si usás:

- `AIDEEN_PROGRESS_EVERY=0`
- `AIDEEN_VAL_EVERY=0`

el proceso puede estar vivo pero no imprimir nada útil por bastante tiempo.

En ese caso parece colgado, pero no necesariamente lo está.

Si querés ver progreso y TPS honestos durante el run:
- dejar `AIDEEN_PROGRESS_EVERY>0`
- usar el perfil de reporting o uno equivalente

---

### 3. Tokenización / cache

En modo archivo grande:

1. intenta leer el corpus
2. inicializa tokenizer
3. genera o reutiliza `*.tokens.bin`
4. recién después entra al training

Si no existe la cache, puede tardar mucho antes del primer progreso.

### Señales sanas de arranque

Hay que ver líneas como:

- `Modo: archivo grande → ...`
- `Tokenizer: BPE (...)`
- `Cache OK: reutilizando ...tokens.bin`
- `Backend: GPU (Metal)`

Si eso aparece, el training arrancó bien.

---

### 4. Launch en background no confiable

En este entorno, lanzar con `nohup ... &` resultó menos confiable que correr en una sesión
TTY persistente. Hubo casos donde el proceso moría antes de escribir al log.

### Recomendación operativa

Para runs importantes:
- lanzarlo en una sesión terminal viva
- o en tmux/screen si aplica

No asumir que background + log vacío significa necesariamente que el trainer está roto;
varias veces fue el launch, no el training.

---

### 5. Branch / árbol contaminado

También vimos que traer cambios core mezclados desde otro PR puede alterar:

- semántica del DEQ
- scratch layout
- defaults de history
- forward shader

Eso puede dar síntomas de:
- cuelgue aparente
- TPS absurdos
- comportamiento inconsistente

Antes de culpar al modelo, verificar que la base core esté limpia.

---

## 6. Qué cosas se podrían evitar

Sí, varias cosas deberían poder evitarse después.

### Se pueden evitar

1. **Comandos demasiado sensibles**
- deberíamos tener wrappers más claros y menos env vars obligatorias

2. **Launch ambiguity**
- idealmente un script oficial de training real y otro de reporting

3. **Pérdida de visibilidad**
- no deberíamos necesitar recordar manualmente cuándo la loss está observándose y cuándo no

4. **Confusión entre throughput y reporting**
- debería haber perfiles oficiales y simples, no adivinados

5. **Dependencia fuerte del estado del árbol**
- necesitamos menos riesgo de mezclar cambios core con runs de benchmark

### No se pueden evitar del todo, por ahora

1. **distintos perfiles para distintos objetivos**
- throughput máximo y reporting de calidad siguen siendo objetivos distintos

2. **tokenización/caché inicial**
- el corpus grande siempre va a tener un costo inicial

3. **cierta sensibilidad a hardware/backend**
- Metal / WGSL / subgroup / portable path siguen importando

---

## 7. Reglas operativas para no repetir errores

1. No usar el bin `train` con un `train` extra en los args.
2. Para corpus grande, usar ruta absoluta al archivo.
3. Para throughput real, usar el comando estable de este documento.
4. Para calidad/reporting, usar el perfil de reporting.
5. Para inspección de checkpoint, usar `infer` o `report_checkpoint.sh`.
6. No interpretar `loss=0.0000` como métrica real si el perfil está sin readback/val.
7. No comparar TPS temprana de warm-up con `tps_epoch` sostenida.

---

## 8. Comandos oficiales

### Training real estable

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=model_large \
  AIDEEN_BATCH_SIZE=8 \
  AIDEEN_CTX_LEN=512 \
  AIDEEN_LM_FUSED_B19=1 \
  AIDEEN_DEQ_HIST_GATED=1 \
  AIDEEN_LOSS_READBACK_EVERY=0 \
  AIDEEN_TPS_SYNC_EVERY=0 \
  AIDEEN_VAL_EVERY=0 \
  AIDEEN_PROGRESS_EVERY=20 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/aideen/corpus_combined.txt \
    --epochs 12 \
    --log-every 1 \
    --save-every 0
```

### Escalera de aprendizaje `tiny -> corpus`

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./train_learning_ladder.sh both
```

Qué hace:
- stage 1: entrena corto sobre `tinyshakespeare` y guarda `${BASE}_tiny`
- stage 2: reanuda desde `${BASE}_tiny` sobre `corpus_combined.txt` y guarda `${BASE}_corpus`
- al final de cada stage corre `report_checkpoint.sh`

Variables útiles:
- `AIDEEN_LADDER_BASE=model_ladder`
- `AIDEEN_TINY_EPOCHS=1`
- `AIDEEN_CORPUS_EPOCHS=1`
- `AIDEEN_MAX_CHUNKS=40` para acotar el stage tiny
- `AIDEEN_MAX_CHUNKS_CORPUS=40` para smoke test corto del stage corpus

### Training con reporting

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./benchmark_fused_profiles.sh report
```

### Training con reporting sin history

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./benchmark_fused_profiles.sh report-nohist
```

### Inferencia rápida

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
cargo run --release --features wgpu -p aideen-training --bin infer -- \
  --model model_large \
  --prompt "The Rust Programming Language is" \
  --max-tokens 48 \
  --temperature 0.15 \
  --top-p 0.75 \
  --top-k 6 \
  --rep-penalty 1.15
```

### Reporte de checkpoint

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./report_checkpoint.sh model_large
```

### Resume estable

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
AIDEEN_RESUME_BASE=model_large AIDEEN_CHECKPOINT_BASE=model_large_resume ./resume_training.sh 3
```

---

## 9. Criterio práctico

### Si querés saber “¿está corriendo?”
Mirar:
- arranque del corpus
- tokenizer
- cache
- backend GPU
- primer `[progress]`

### Si querés saber “¿está entrenando bien?”
No usar el perfil throughput puro.
Usar:
- reporting
- loss visible
- checkpoint report

### Si querés saber “¿el modelo ya aprendió?”
No alcanza con que el training corra.
Hay que mirar:
- `loss`
- `tps_epoch`
- `report_checkpoint.sh`
- calidad de salida en prompts fijos

---

## 10. Próxima simplificación recomendada

Cuando baje un poco la intensidad del período de análisis, conviene hacer esto:

1. unificar training real en un script oficial único
2. unificar reporting en otro script oficial único
3. dejar un solo lugar donde viven los defaults operativos
4. evitar depender de una constelación de env vars para tareas comunes

Ese trabajo no reemplaza el análisis del modelo, pero sí reduce mucho el ruido operativo.
