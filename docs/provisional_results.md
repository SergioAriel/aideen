### [2026-03-20] Baseline provisional: V dinámica estable (post V-normalization fix)
**Contexto**: estabilidad DEQ con V dinámica y shader estabilizado.
**Objetivo**: eliminar DEQ-INVALID y mantener contr < 1 sin fijar V.

**Configuración exacta**
- Cmd: `<no registrado — run desde IDE>`
- Env: `<no registrado>`
- Seed: 7 / 11 / 13 / 42
- Iters: 100
- Dataset: stress_test
- Perfil: STABLE
- Código: cambios post “V-normalization fix” (archivos no listados en logs)

**Resultados**
- mode/conv: NORMAL / OK
- contr/maxΔ: contr≈0.20 (antes 1.66), iters≈7 (antes 46)
- hist/inj: ≈0.079 estable
- DEQ-INVALID: 0 en los 4 seeds

**Criterios de invalidez**
- Si reaparece DEQ-INVALID en cualquiera de esos seeds bajo misma config.
- Si cambia el shader DEQ o la normalización de V, revalidar.
- Si no se puede reproducir con un comando explícito, esta entrada se considera provisional no verificada.

**Alcance**
- Válido solo para stress_test y estas semillas; no implica default.

### [2026-04-10] Baseline provisional: FPM stage 4 estable para training auditado
**Contexto**: estabilización de `stage=4` en el camino FPM, cerrando dos frentes a la vez:
memoria de escritura (`H -> M`) y observabilidad confiable del `debug_buf`.
**Objetivo**: dejar un baseline estable para volver a entrenar sin `DEQ-INVALID` falsos ni
telemetría corrupta, y con localización del token interno de los máximos de solve/attention.

**Configuración exacta**
- Cmd: `AIDEEN_CTX_LEN=512 AIDEEN_BATCH_SIZE=1 AIDEEN_DEBUG_SAMPLE=10 AIDEEN_DEBUG_FPM=1 AIDEEN_FPM_STAGE=4 cargo run --features wgpu -p aideen-training --bin train --release -- --file corpus_pg19_train.txt --epochs 1 --log-every 10 --freeze-deq --freeze-emb --freeze-lm`
- Smoke: `AIDEEN_CTX_LEN=512 AIDEEN_BATCH_SIZE=1 AIDEEN_DEBUG_SAMPLE=10 AIDEEN_DEBUG_FPM=1 AIDEEN_FPM_STAGE=4 cargo run --features wgpu -p aideen-training --bin train --release -- --file corpus_pg19_train_smoke.txt --epochs 1 --log-every 10 --freeze-deq --freeze-emb --freeze-lm`
- Env: `AIDEEN_CTX_LEN=512`, `AIDEEN_BATCH_SIZE=1`, `AIDEEN_DEBUG_SAMPLE=10`, `AIDEEN_DEBUG_FPM=1`, `AIDEEN_FPM_STAGE=4`
- Seed: no fijada explícitamente
- Iters: corrida larga auditada hasta al menos `step 100` y smoke adicional
- Dataset: `corpus_pg19_train.txt` y `corpus_pg19_train_smoke.txt`
- Perfil: `EMERG/NORMAL` según controlador adaptivo; `conv=OK`
- Código:
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl`
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-training-lab/src/trainer.rs`

**Resultados**
- mode/conv: `conv=OK` en la corrida larga auditada; sin `NaN`, `panic`, `INV` ni `DEQ-INVALID`
- solve/contr:
  - `step 10`: `contr=1.151`, `solve=1.332e-1`
  - `step 20`: `contr=1.934`, `solve=3.678e-1`
  - `step 30`: `contr=1.363`, `solve=4.530e-1`
  - `step 40`: `contr=1.140`, `solve=2.899e-1`
  - `step 50`: `contr=1.551`, `solve=2.391e-1`
  - `step 60`: `contr=1.827`, `solve=3.121e-1`
  - `step 70`: `contr=1.758`, `solve=2.134e-1`
  - `step 80`: `contr=1.826`, `solve=5.897e-1`
  - `step 90`: `contr=1.687`, `solve=2.175e-1`
  - `step 100`: `conv=OK` mantenido en la continuación auditada
- throughput: banda observada ~`160-182 TPS` en la corrida larga congelada
- memoria:
  - `err_M` quedó alrededor de `1.0-1.23`
  - `z_avg≈0.182`, `z_max≈0.193`
  - `memctx/sig≈1.20e-3` a `1.30e-3`
- token localization:
  - `step 30`: `imax_err@tok=413`, `imax_a@tok=413`
  - `step 50`: `imax_err@tok=357`, `imax_a@tok=357`
  - los máximos aparecen en tokens internos del bloque, no solo al inicio
- fix estructural asociado:
  - el carrier `H -> M` usa `sqrt(write_budget)` en la escritura
  - el `trainer` ya no promueve snapshots intermedios (`sig=186`) al `cached_debug_buf`

**Criterios de invalidez**
- Si reaparecen snapshots inválidos persistentes (`sig!=901`) que vuelvan a contaminar el control loop, esta entrada queda invalidada.
- Si con la misma configuración reaparecen `DEQ-INVALID`, `INV`, `panic` o una explosión comparable al spike histórico en `step 50`, revalidar.
- Si cambia cualquiera de estos archivos, revalidar:
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl`
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-training-lab/src/trainer.rs`
- Si se cambia la ecuación de write (`H -> M`) o el lifecycle del debug snapshot, esta baseline deja de aplicar.

**Alcance**
- Válido para auditoría y arranque de training bajo `stage=4` con esta configuración congelada.
- No implica que sea el default final ni que la calidad de aprendizaje ya esté cerrada.
- Es un baseline de estabilidad operativa y observabilidad confiable, no una validación final de calidad.
