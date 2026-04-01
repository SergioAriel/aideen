# Training Incident Log

## Incident #1 — GPU Context Loss with ctx_len=512

- **Date:** 2026-03-24 18:36 UTC-3
- **Duration:** ~2 minutes before hang
- **Command:** `AIDEEN_CTX_LEN=512 cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus_fineweb_50m.txt --resume model_large --epochs 1 --log-every 10 --save-every 1`
- **Error:** `radv/amdgpu: The CS has been cancelled because the context is lost. This context is guilty of a soft recovery.`
- **Impact:** No data loss — checkpoint was not modified.

## Incident #2 — GPU hangs persist across all training modes

- **Date:** 2026-03-24 18:40-19:50 UTC-3
- **Attempts:** 5+ (various ctx_len, corpus, TTY vs desktop, 4 reboots)
- **Error:** All crash at or shortly after first training step with `radv/amdgpu: The CS has been cancelled because the context is lost.`

## Incident #3 — GPU clock scaling (partial cause)

- **Date:** 2026-03-24 ~20:00 UTC-3
- **Hypothesis:** AMD Radeon 780M (iGPU) `power_dpm_force_performance_level = auto` starts at 800 MHz (minimum). The DEQ+Picard compute shaders (~6 iterations, multiple dispatches) exceed the amdgpu driver's fixed ring timeout at low clocks.
- **Why it worked before:** The successful 12-hour training (model_large.aidn, completed 09:33) ran during a 6-day uptime boot (Mar 18-24). Sustained GPU load had already pushed clocks to 2799 MHz. After rebooting, the GPU resets to 800 MHz.
- **Status:** PARTIAL FIX ONLY — see Incident #4.

## Incident #4 — Clocks alone are NOT enough; driver state matters

- **Date:** 2026-03-24 ~21:00 UTC-3
- **What happened:** After forcing `high` (confirmed 2799 MHz via pp_dpm_sclk), training STILL crashed with `context is lost` on the first compute dispatch (validation chunk 0).
- **Root cause (revised):** TWO factors must be satisfied:
  1. **GPU clocks must be at max** (high) — not auto/800 MHz
  2. **Driver must be in a clean state** — after multiple soft recoveries, the amdgpu driver marks the GPU context as "guilty" and subsequent dispatches fail even at full clock speed. The driver accumulates corruption from repeated `context is lost` errors.
- **Why the 12-hour training worked:** That session ran on a fresh boot (Mar 18) with 6 days of uptime. The GPU had never experienced a soft recovery in that boot cycle, AND clocks were already high from sustained load.
- **Evidence:**
  - `pp_dpm_sclk` confirmed `2: 2799Mhz *` at time of crash
  - `pp_dpm_mclk` confirmed `2: 2800Mhz *`
  - `gpu_busy_percent` was 3% (not overloaded)
  - Multiple prior soft recoveries in this boot cycle from failed attempts at 800 MHz
  - No code changes, no package updates since Mar 14
  - VRAM (2 GB) not the limiting factor — crash occurs from TTY without compositor
  - Closing desktop environment made no difference
- **Fix:** The launch procedure MUST be: reboot -> force high -> train (in that order, no failed attempts in between).

## Updated Launch Procedure (CRITICAL ORDER)

**After every reboot, do this IMMEDIATELY before anything else:**

1. Reboot the machine
2. Force GPU to max clocks FIRST (before any GPU workload):
   ```bash
   echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
   cat /sys/class/drm/card1/device/pp_dpm_sclk  # verify 2799Mhz *
   ```
3. Launch training (FIRST GPU workload after reboot):
   ```bash
   bash ~/sustainability/aideen/run_training.sh
   ```
4. Wait 2-3 min for `[progress]` lines to confirm training is running
5. Monitor: `tail -f ~/sustainability/aideen/training_full_*.log`
6. Training duration: ~88 hours (1 epoch of corpus_combined, 3.76M tokens)
7. After training: run benchmark, write report, update grant proposal

**IMPORTANT RULES:**
- If training crashes, DO NOT retry without rebooting first. Each crash poisons the driver state.
- The `high` setting does not survive reboots — must be re-applied every time.
- Do NOT attempt training at `auto`/800 MHz even "just to test" — it will crash and contaminate the driver.
- The correct order is always: fresh reboot -> force high -> launch training.

**Optional:** To avoid forgetting step 2, create a systemd service that runs `echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level` at boot.
