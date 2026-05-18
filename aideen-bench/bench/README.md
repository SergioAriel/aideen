# AIDEEN Bench Matrix

Fixed profiles to compare stability/convergence of DEQ training on GPU.

## Files

- `profiles.csv`: ladder of configurations (`512 -> 1024`)
- `run_matrix.sh`: runs `stress_test` per profile and consolidates metrics

## Run

From the repo root:

```bash
./aideen-bench/bench/run_matrix.sh
```

Optional:

```bash
./aideen-bench/bench/run_matrix.sh ./aideen-bench/bench/profiles.csv ./aideen-bench/bench/results/manual_run
```

## Output

Generates:

- logs per profile: `results/<timestamp>/<profile>.log`
- summary table: `results/<timestamp>/summary.csv`

Key columns:

- `iter20_loss`
- `hit_ratio_pct`
- `conv`
- `mode`
- `contractivity`
- `max_delta`
- `rs_cg`
- `shared`

## Promotion criterion

Promote to the next `d_r` only if:

- `conv=OK`
- `hit_ratio_pct <= 5`
- `mode=NORMAL` for most of the run
- `contractivity <= 1.0` without sustained spikes

## v2 profile (convergence tuning)

To try to move `d_r=768` to `conv=OK`:

```bash
./aideen-bench/bench/run_matrix.sh ./aideen-bench/bench/profiles_v2.csv
```
