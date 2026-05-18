### [2026-03-20] Provisional baseline: stable dynamic V (post V-normalization fix)
**Context**: DEQ stability with dynamic V and a stabilized shader.
**Goal**: eliminate DEQ-INVALID and keep contr < 1 without fixing V.

**Exact configuration**
- Cmd: `<not recorded — run from IDE>`
- Env: `<not recorded>`
- Seed: 7 / 11 / 13 / 42
- Iters: 100
- Dataset: stress_test
- Profile: STABLE
- Code: changes after the "V-normalization fix" (files not listed in logs)

**Results**
- mode/conv: NORMAL / OK
- contr/maxΔ: contr≈0.20 (previously 1.66), iters≈7 (previously 46)
- hist/inj: ≈0.079 stable
- DEQ-INVALID: 0 across all 4 seeds

**Invalidation criteria**
- If DEQ-INVALID reappears in any of those seeds under the same config.
- If the DEQ shader or the V normalization changes, revalidate.
- If it cannot be reproduced with an explicit command, this entry is considered provisional and unverified.

**Scope**
- Valid only for stress_test and these seeds; it does not imply a default.

### [2026-04-10] Provisional baseline: FPM stage 4 stable for audited training
**Context**: stabilization of `stage=4` in the FPM path, closing two fronts at once:
write memory (`H -> M`) and reliable observability of the `debug_buf`.
**Goal**: leave a stable baseline to resume training without false `DEQ-INVALID` or
corrupt telemetry, and with localization of the internal token of the solve/attention maxima.

**Exact configuration**
- Cmd: `AIDEEN_CTX_LEN=512 AIDEEN_BATCH_SIZE=1 AIDEEN_DEBUG_SAMPLE=10 AIDEEN_DEBUG_FPM=1 AIDEEN_FPM_STAGE=4 cargo run --features wgpu -p aideen-training --bin train --release -- --file corpus_pg19_train.txt --epochs 1 --log-every 10 --freeze-deq --freeze-emb --freeze-lm`
- Smoke: `AIDEEN_CTX_LEN=512 AIDEEN_BATCH_SIZE=1 AIDEEN_DEBUG_SAMPLE=10 AIDEEN_DEBUG_FPM=1 AIDEEN_FPM_STAGE=4 cargo run --features wgpu -p aideen-training --bin train --release -- --file corpus_pg19_train_smoke.txt --epochs 1 --log-every 10 --freeze-deq --freeze-emb --freeze-lm`
- Env: `AIDEEN_CTX_LEN=512`, `AIDEEN_BATCH_SIZE=1`, `AIDEEN_DEBUG_SAMPLE=10`, `AIDEEN_DEBUG_FPM=1`, `AIDEEN_FPM_STAGE=4`
- Seed: not explicitly fixed
- Iters: long audited run up to at least `step 100` plus an additional smoke
- Dataset: `corpus_pg19_train.txt` and `corpus_pg19_train_smoke.txt`
- Profile: `EMERG/NORMAL` according to the adaptive controller; `conv=OK`
- Code:
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl`
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-training-lab/src/trainer.rs`

**Results**
- mode/conv: `conv=OK` in the long audited run; no `NaN`, `panic`, `INV` or `DEQ-INVALID`
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
  - `step 100`: `conv=OK` maintained in the audited continuation
- throughput: observed band ~`160-182 TPS` in the long frozen run
- memory:
  - `err_M` stayed around `1.0-1.23`
  - `z_avg≈0.182`, `z_max≈0.193`
  - `memctx/sig≈1.20e-3` to `1.30e-3`
- token localization:
  - `step 30`: `imax_err@tok=413`, `imax_a@tok=413`
  - `step 50`: `imax_err@tok=357`, `imax_a@tok=357`
  - the maxima appear at internal tokens of the block, not only at the start
- associated structural fix:
  - the `H -> M` carrier uses `sqrt(write_budget)` in the write
  - the `trainer` no longer promotes intermediate snapshots (`sig=186`) to the `cached_debug_buf`

**Invalidation criteria**
- If persistent invalid snapshots (`sig!=901`) reappear and contaminate the control loop again, this entry is invalidated.
- If with the same configuration `DEQ-INVALID`, `INV`, `panic` reappear or an explosion comparable to the historical spike at `step 50` occurs, revalidate.
- If any of these files changes, revalidate:
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl`
  - `/Users/sergiosolis/Programacion/AIDEEN/aideen-training-lab/src/trainer.rs`
- If the write equation (`H -> M`) or the debug snapshot lifecycle changes, this baseline no longer applies.

**Scope**
- Valid for auditing and starting training under `stage=4` with this frozen configuration.
- It does not imply this is the final default nor that learning quality is already settled.
- It is a baseline of operational stability and reliable observability, not a final quality validation.
