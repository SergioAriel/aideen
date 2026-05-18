# Training Operations Guide

Operational guide for running training, reporting and inference in AIDEEN without falling into
the problems we already saw in this round: malformed commands, lack of visibility,
misleading TPS and runs that look "hung" when in reality they are tokenizing,
compiling or waiting for a first progress milestone.

## Objective

This document does not try to freeze the system forever. AIDEEN is still in an
analysis and improvement phase. But it does fix a stable operational base for:

1. launching real training with a known configuration;
2. obtaining visible and comparable metrics;
3. running inference over checkpoints without depending on the interactive chat;
4. reducing operational errors that should not happen again.

---

## Current state of the system

### What is already stabilized

- The real `train` runs fine with the current fused path.
- The benchmark/runner already distinguishes throughput vs reporting profiles.
- Progress now differentiates:
  - `tps_win`: window throughput
  - `tps_run`: accumulated run throughput
- When `AIDEEN_PROGRESS_EVERY>0`, those TPS already measure completed GPU work.
  The trainer syncs at each progress cut to avoid numbers inflated by queued commands.
- If there is no reliable visible loss, the trainer shows `loss=n/a` instead of `0.0000`.
- Fast inference now has a dedicated bin:
  - `aideen-training-lab/src/bin/infer.rs`
- Checkpoint inspection now has a dedicated script:
  - `report_checkpoint.sh`

### What is still debt or a limitation

- Some defaults are still too sensitive to the usage profile.
- Background launch via `nohup &` was unreliable in this environment; the persistent TTY
  session worked better.
- The current `model_large` checkpoint still does not generate useful text.
- The history branch exists, but the observed checkpoint showed `hist_scale` almost off.

---

## Main rule

Do not use a single command for everything.

In AIDEEN today there are three different tasks:

1. **real throughput training**
2. **training with visible reporting/quality**
3. **inference / checkpoint evaluation**

Each one requires a distinct profile.

---

## 1. Stable real training

This is the recommended base command today for real pretraining over the clean corpus.

### Command

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=/Users/sergiosolis/Programacion/AIDEEN/artifacts/checkpoints/model_histv2_clean_pretrain_latest \
  AIDEEN_BATCH_SIZE=4 \
  AIDEEN_CTX_LEN=512 \
  AIDEEN_DEQ_TOKEN_CARRY=1 \
  AIDEEN_ADJ_ITERS_OVERRIDE=2 \
  AIDEEN_LOSS_READBACK_EVERY=0 \
  AIDEEN_TPS_SYNC_EVERY=0 \
  AIDEEN_VAL_EVERY=0 \
  AIDEEN_PROGRESS_EVERY=0 \
  AIDEEN_MAX_CHUNKS=160 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/AIDEEN/corpus_pretrain_minimal.txt \
    --epochs 2 \
    --log-every 1 \
    --save-every 1
```

### What each important flag does

- `AIDEEN_BATCH_SIZE=4`
  - profile validated for M1 Pro on the canonical `unified` path
- `AIDEEN_CTX_LEN=512`
  - better occupancy than `256` in this regime
- `AIDEEN_DEQ_TOKEN_CARRY=1`
  - keeps `H_curr` active as the DEQ baseline state/carry
- current canonical path
  - `slot_ctx = Attn(signal)`
  - `pre = signal + H_curr + slot_ctx + slot_anchor`
  - the explicit history `HistCtx/MState` is out of the baseline for now
- `AIDEEN_ADJ_ITERS_OVERRIDE=2`
  - fixes the adjoint for clean comparisons and prevents the per-epoch scheduler from contaminating TPS/loss
- `AIDEEN_LOSS_READBACK_EVERY=0`
  - do not block training with intra-step loss readbacks
- `AIDEEN_TPS_SYNC_EVERY=0`
  - do not put observability syncs in the hot path
- `AIDEEN_VAL_EVERY=0`
  - do not validate during throughput training
- `AIDEEN_PROGRESS_EVERY=0`
  - avoids extra syncs in this stable throughput recipe
- `AIDEEN_MAX_CHUNKS=160`
  - minimal serious recipe that already showed a real improvement on the clean corpus
- `AIDEEN_CHECKPOINT_BASE=.../model_histv2_clean_pretrain_latest`
  - saves to `artifacts/checkpoints/` and avoids dirtying the repo root
  - it now also generates:
    - `..._metrics.csv`
    - `..._best_loss.aidn/.opt`
    - `..._best_loss.meta`

### When to use it

- real runs over the clean corpus
- real throughput with the current history
- training of the main checkpoint

### What not to expect

- useful loss on every chunk
- intermediate `val_loss`
- fine-grained quality diagnostics

This profile is optimized to **run** in the current good regime, not to compare variants.

---

## Recommended corpus for serious tests

Do not use `corpus_combined.txt` directly as the main pretraining corpus.

The audit showed that this file ended up dominated by blocks:

- `USER:`
- `ASSISTANT:`
- `SYSTEM:`

That is useful for an instruct/chat stage, but not for the model's base
pretraining.

### Generate clean and separate corpora

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
python3 prepare_training_corpora.py
```

This generates in the current workspace:

- `corpus_pretrain_minimal.txt`
- `corpus_chat_instruct.txt`

### What each one contains

- `corpus_pretrain_minimal.txt`
  - Rust Book
  - clean FineWeb extracted from the combined corpus
  - local project documentation:
    - `README.md`
    - `ARCHITECTURE.md`
    - `PLAN.md`
    - `ARCHITECTURE_DECISIONS.md`
    - `docs/distributed_training_users.md`

- `corpus_chat_instruct.txt`
  - the `USER:/ASSISTANT:/SYSTEM:` conversational block separated for a
    future finetune/chat stage

### Recommended smoke test

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
AIDEEN_LADDER_BASE=model_clean_probe \
AIDEEN_CORPUS_FILE=/Users/sergiosolis/Programacion/AIDEEN/corpus_pretrain_minimal.txt \
AIDEEN_TINY_EPOCHS=1 \
AIDEEN_CORPUS_EPOCHS=1 \
AIDEEN_MAX_CHUNKS=40 \
AIDEEN_MAX_CHUNKS_CORPUS=40 \
./train_learning_ladder.sh both
```

This run does not aim for final quality. It aims to validate that:

1. the model learns a more coherent distribution;
2. degenerate tokens like `ASS/USER/IST/ANT` do not reappear;
3. the local carry `H_curr` stays stable without the corpus burying it.

### Important invariant

`train_on_file` resets the temporal state at the start of each epoch.

That is still mandatory because the file returns to the beginning on each epoch and
`H_curr` cannot carry state from the end of the previous epoch.

---

## 2. Training with visible reporting

When the goal is to evaluate quality/learning and not just throughput, use the reporting
profile.

### Current minimal serious runner

This is the recommended runner for comparing stabilization changes before
promoting them. It keeps the scalable Assoc banks active, but leaves the
unvalidated bridges to FPM and LM disabled.

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=model_serious_prep \
  AIDEEN_TRAIN_SEED=42 \
  AIDEEN_LM_SAMPLE_SEED=42 \
  AIDEEN_BATCH_SIZE=4 \
  AIDEEN_CTX_LEN=256 \
  AIDEEN_H_SLOTS=8 \
  AIDEEN_ASSOC_BANKS=32 \
  AIDEEN_ASSOC_WRITE_BUDGET=32 \
  AIDEEN_ASSOC_SLOT_OWNER=1 \
  AIDEEN_ASSOC_HASH_LANE=1 \
  AIDEEN_ASSOC_HASH_REPLICA=1 \
  AIDEEN_ASSOC_EVENT_GATE=1 \
  AIDEEN_ASSOC_SALIENCE_REPLACE=1 \
  AIDEEN_ASSOC_CONF_READ=1 \
  AIDEEN_ASSOC_TO_FPM_SCALE=0.0 \
  AIDEEN_ASSOC_LOGIT_LAMBDA=0.0 \
  AIDEEN_HELDOUT_MAX_TOKENS=2048 \
  AIDEEN_MAX_CHUNKS=20 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/AIDEEN/corpus_pretrain_minimal.txt \
    --val-file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
    --epochs 1 \
    --log-every 1 \
    --save-every 0
```

Recorded smoke baseline:

- `logs/real_text/pretrain_minimal_heldout_tiny_smoke_chunks20.log`
- train loss `6.7353`
- held-out `val_loss=6.7181`
- `135.7 TPS`
- no `NaN` / `DEQ-INVALID`

To move from smoke to serious training, first raise `AIDEEN_MAX_CHUNKS` and then
`AIDEEN_HELDOUT_MAX_TOKENS`, keeping those bridges at `0.0` until a
variant beats them on held-out and TPS.

### Recommended command

```bash
cd /Users/sergiosolis/Programacion/AIDEEN && \
env \
  AIDEEN_CHECKPOINT_BASE=model_report \
  AIDEEN_BATCH_SIZE=4 \
  AIDEEN_CTX_LEN=512 \
  AIDEEN_DEQ_TOKEN_CARRY=0 \
  AIDEEN_ADJ_ITERS_OVERRIDE=2 \
  AIDEEN_LOSS_READBACK_EVERY=20 \
  AIDEEN_TPS_SYNC_EVERY=20 \
  AIDEEN_VAL_EVERY=200 \
  AIDEEN_PROGRESS_EVERY=20 \
  AIDEEN_MAX_CHUNKS=200 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/AIDEEN/corpus_pretrain_minimal.txt \
    --epochs 1 \
    --log-every 1 \
    --save-every 0
```

### When to use it

- measure visible `loss`
- compare `history on/off`
- validate that a modification really improves training
- short and comparable runs

### What to expect

- progress with `loss=...` or an honest `loss=n/a`
- `tps_win`
- `tps_run`
- `tps_epoch` at the end
- a separate checkpoint if you enable `--save-every > 0`
- persistent per-epoch metrics in `AIDEEN_CHECKPOINT_BASE_metrics.csv`
- automatic `best_loss` checkpoint without having to pick epochs by hand

---

## 3. Fast inference with a checkpoint

Use the `infer` bin.

### Basic command

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

### With checkpoint stats

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

### What it reports

- load time
- generation time
- `tok/s`
- stats for:
  - embeddings
  - LM head
  - history params

---

## 4. Reproducible checkpoint report

Use the script:

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./report_checkpoint.sh model_large
```

### What it does

- runs inference with `history on`
- runs inference with `history off`
- uses fixed prompts
- prints checkpoint stats

### What it is for

- comparing training branches
- inspecting whether history is alive
- seeing whether the model is still in "token soup"

---

## 5. Typical causes of "looks hung"

These were real causes observed.

### 1. Malformed command

The `train` bin does **not** use a `train` subcommand.

### Correct

```bash
cargo run --release --features wgpu -p aideen-training --bin train -- --file ...
```

### Incorrect

```bash
cargo run --release --features wgpu -p aideen-training --bin train -- train --file ...
```

That extra `train` can make the process die or make the launch fail before starting properly.

---

### 2. No visible progress

If you use:

- `AIDEEN_PROGRESS_EVERY=0`
- `AIDEEN_VAL_EVERY=0`

the process can be alive but print nothing useful for quite a while.

In that case it looks hung, but it is not necessarily so.

If you want to see honest progress and TPS during the run:
- keep `AIDEEN_PROGRESS_EVERY>0`
- use the reporting profile or an equivalent one

---

### 3. Tokenization / cache

In large file mode:

1. it tries to read the corpus
2. it initializes the tokenizer
3. it generates or reuses `*.tokens.bin`
4. only then does it enter training

If the cache does not exist, it can take a long time before the first progress.

### Healthy startup signs

You should see lines like:

- `Mode: large file → ...`
- `Tokenizer: BPE (...)`
- `Cache OK: reusing ...tokens.bin`
- `Backend: GPU (Metal)`

If that appears, the training started fine.

---

### 4. Unreliable background launch

In this environment, launching with `nohup ... &` turned out less reliable than running in a persistent
TTY session. There were cases where the process died before writing to the log.

### Operational recommendation

For important runs:
- launch it in a live terminal session
- or in tmux/screen if applicable

Do not assume that background + empty log necessarily means the trainer is broken;
several times it was the launch, not the training.

---

### 5. Contaminated branch / tree

We also saw that pulling mixed core changes from another PR can alter:

- DEQ semantics
- scratch layout
- history defaults
- forward shader

That can produce symptoms of:
- apparent hang
- absurd TPS
- inconsistent behavior

Before blaming the model, verify that the core base is clean.

---

## 6. What could be avoided

Yes, several things should be avoidable later.

### Can be avoided

1. **Commands that are too sensitive**
- we should have clearer wrappers and fewer mandatory env vars

2. **Launch ambiguity**
- ideally one official script for real training and another for reporting

3. **Loss of visibility**
- we should not need to manually remember when the loss is being observed and when not

4. **Confusion between throughput and reporting**
- there should be official and simple profiles, not guessed ones

5. **Strong dependence on tree state**
- we need less risk of mixing core changes with benchmark runs

### Cannot be fully avoided, for now

1. **distinct profiles for distinct goals**
- maximum throughput and quality reporting remain distinct goals

2. **initial tokenization/cache**
- the large corpus will always have an initial cost

3. **some hardware/backend sensitivity**
- Metal / WGSL / subgroup / portable path still matter

---

## 7. Operational rules to avoid repeating errors

1. Do not use the `train` bin with an extra `train` in the args.
2. For a large corpus, use an absolute path to the file.
3. For real throughput, use the stable command in this document.
4. For quality/reporting, use the reporting profile.
5. For checkpoint inspection, use `infer` or `report_checkpoint.sh`.
6. Do not interpret `loss=0.0000` as a real metric if the profile has no readback/val.
7. Do not compare early warm-up TPS with sustained `tps_epoch`.

---

## 8. Official commands

### Stable real training

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

### Learning ladder `tiny -> corpus`

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./train_learning_ladder.sh both
```

What it does:
- stage 1: trains briefly on `tinyshakespeare` and saves `${BASE}_tiny`
- stage 2: resumes from `${BASE}_tiny` on `corpus_combined.txt` and saves `${BASE}_corpus`
- at the end of each stage it runs `report_checkpoint.sh`

Useful variables:
- `AIDEEN_LADDER_BASE=model_ladder`
- `AIDEEN_TINY_EPOCHS=1`
- `AIDEEN_CORPUS_EPOCHS=1`
- `AIDEEN_MAX_CHUNKS=40` to bound the tiny stage
- `AIDEEN_MAX_CHUNKS_CORPUS=40` for a short smoke test of the corpus stage

### Training with reporting

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./benchmark_fused_profiles.sh report
```

### Training with reporting, no history

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./benchmark_fused_profiles.sh report-nohist
```

### Fast inference

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

### Checkpoint report

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
./report_checkpoint.sh model_large
```

### Stable resume

For continuation probes, `AIDEEN_LR` also applies with `--resume`.
Use an explicit and small value to resume from an already-trained checkpoint.
In addition, the `train_on_file` scheduler now continues by the checkpoint's
global epochs: when resuming it does not start over at the high LR of the first epoch of the
new run.

### LR self-regulation at runtime

`train_on_file` now includes a persistent plateau controller:

- it measures the effective `loss` of the epoch
- it keeps the run's internal `best_loss`
- it reduces the `lr_cap` when there is a sustained plateau
- it persists that state in the checkpoint

Current defaults:

- patience: `2` epochs
- cooldown: `1` epoch
- factor: `0.5`
- minimum relative improvement: `0.005`
- `min_lr`: uses `training_config.lr_min` unless overridden

Available overrides:

- `AIDEEN_LR_PLATEAU_DISABLE=1`
- `AIDEEN_LR_PLATEAU_PATIENCE=<n>`
- `AIDEEN_LR_PLATEAU_COOLDOWN=<n>`
- `AIDEEN_LR_PLATEAU_FACTOR=<f>`
- `AIDEEN_LR_PLATEAU_MIN_REL_IMPROVEMENT=<f>`
- `AIDEEN_LR_PLATEAU_MIN_LR=<lr>`

Previous provisional baseline validated for continuation over
`artifacts/checkpoints/model_histv2_clean_pretrain_latest`:

- `AIDEEN_LR=0.00002`
- `AIDEEN_MAX_CHUNKS=160`
- `AIDEEN_ADJ_ITERS_OVERRIDE=2`
- `AIDEEN_DEQ_TOKEN_CARRY=0`

In this configuration, the continuation probes ended in a useful `loss` band of
high-7 / low-8, clearly better than the previous resume regime that
had degraded down to ~`8.0-9.9`.

Minimal operational validation of the continuous schedule:

- fresh run of `1` epoch: `lr=0.000020`
- immediate resume for `1` epoch over the same checkpoint: `lr=0.000011`

That is: the resume no longer restarts the cosine schedule from scratch.

Minimal operational validation of the plateau controller:

- forced probe of `3` epochs with:
  - `AIDEEN_LR_PLATEAU_PATIENCE=1`
  - `AIDEEN_LR_PLATEAU_COOLDOWN=0`
  - `AIDEEN_LR_PLATEAU_MIN_REL_IMPROVEMENT=0.05`
- result:
  - epoch 0: `lr=0.000020`
  - epoch 1: plateau detected, `lr_cap -> 0.000005`
  - epoch 2: `lr=0.00000321`

That is: the run can now lower LR within the same run without depending on manual relaunches.

### Fixed-point control

The solve controller no longer depends on `AIDEEN_DEBUG_SAMPLE`.

- `AIDEEN_DEBUG_SAMPLE`
  - controls only human logs (`[GPU-DEBUG]`, `[GPU-HIST]`, `[GPU-SSM]`)
- `AIDEEN_SOLVE_CONTROL_EVERY`
  - controls how many steps apart the solve metrics are sampled to regulate:
    - `adaptive_max_iters`
    - emergency
    - contractivity

Default:

- `AIDEEN_SOLVE_CONTROL_EVERY=10`

This prevents turning off logs from leaving the fixed-point controller dormant.

In addition, the trainer now automatically saves:

- `AIDEEN_CHECKPOINT_BASE.aidn/.opt`
  - latest / operational continuity
- `AIDEEN_CHECKPOINT_BASE_best_loss.aidn/.opt`
  - best epoch observed by the visible effective loss of the epoch
- `AIDEEN_CHECKPOINT_BASE_best_loss.meta`
  - loss, epoch, lr and tokens of the best checkpoint
- `AIDEEN_CHECKPOINT_BASE_metrics.csv`
  - per-epoch series to audit continuity without reading logs by hand

### Current evaluation of `H_currSSM` over `tinyshakespeare`

Canonical recipe to compare the current `H_currSSM` path against the baseline:

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
for seed in 7 11 13 42; do
  echo "=== H_currSSM seed=$seed ==="
  AIDEEN_TRAIN_SEED=$seed \
  AIDEEN_H_HIST=1 \
  AIDEEN_CHECKPOINT_BASE=model_hssm_s$seed \
  AIDEEN_MAX_CHUNKS=100 \
  AIDEEN_PROGRESS_EVERY=20 \
  AIDEEN_LOSS_READBACK_EVERY=10 \
  AIDEEN_BATCH_SIZE=4 \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file /Users/sergiosolis/Programacion/aideen/aideen-bench/tinyshakespeare.txt \
    --epochs 1 --log-every 1 --save-every 0 2>&1 | grep -E "(progress.*chunk|epoch.*loss)"
done
```

Notes:

- `AIDEEN_H_HIST=1` enables the read/write of `H_hist` in the unified shader.
- there is no need to pass `AIDEEN_H_HIST_GAMMA`: the current effective `γ` is obtained from the trainable internal parameter reparameterized in the shader.
- use exactly these four seeds to compare robustness, not a single short run.

```bash
cd /Users/sergiosolis/Programacion/AIDEEN
AIDEEN_RESUME_BASE=/Users/sergiosolis/Programacion/AIDEEN/artifacts/checkpoints/model_histv2_clean_pretrain_latest \
AIDEEN_CHECKPOINT_BASE=/Users/sergiosolis/Programacion/AIDEEN/artifacts/checkpoints/model_histv2_clean_pretrain_latest \
AIDEEN_LR=0.00002 \
./resume_training.sh 3
```

---

## 9. Practical criterion

### If you want to know "is it running?"
Look at:
- corpus startup
- tokenizer
- cache
- GPU backend
- first `[progress]`

### If you want to know "is it training well?"
Do not use the pure throughput profile.
Use:
- reporting
- visible loss
- checkpoint report

### If you want to know "has the model learned yet?"
It is not enough for the training to run.
You have to look at:
- `loss`
- `tps_epoch`
- `report_checkpoint.sh`
- output quality on fixed prompts

---

## 10. Recommended next simplification

When the intensity of the analysis period drops a bit, it is worth doing this:

1. unify real training into a single official script
2. unify reporting into another single official script
3. leave a single place where the operational defaults live
4. avoid depending on a constellation of env vars for common tasks

That work does not replace the model analysis, but it does significantly reduce the operational noise.
