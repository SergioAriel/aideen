#!/bin/bash
# Watchdog for AIDEEN training — checks every hour, relaunches if crashed.
# Usage: nohup bash ~/sustainability/aideen/watchdog_training.sh > ~/sustainability/aideen/watchdog.log 2>&1 &
#
# What it does:
#   1. Checks if the training process is alive every INTERVAL seconds
#   2. If dead: logs the crash, relaunches training
#   3. If alive: logs last progress + val_loss lines
#   4. Repeats until training finishes (all epochs done) or MAX_RESTARTS reached
#
# NOTE: This does NOT handle the GPU driver contamination issue.
#       If training crashes repeatedly, it will stop after MAX_RESTARTS
#       and you'll need to reboot + force GPU high before restarting.

INTERVAL=3600        # check every 1 hour
MAX_RESTARTS=3       # max auto-relaunches before giving up
TRAIN_DIR="$HOME/sustainability/aideen"
TRAIN_CMD="cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus_combined.txt --resume model_large --epochs 1 --log-every 1 --save-every 1"

restarts=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_latest_log() {
    ls -t "$TRAIN_DIR"/training_full_*.log 2>/dev/null | head -1
}

is_training_alive() {
    pgrep -f "target/release/train" > /dev/null 2>&1
}

launch_training() {
    cd "$TRAIN_DIR" || exit 1
    local logfile="training_full_$(date +%Y%m%d_%H%M%S).log"
    nohup $TRAIN_CMD > "$logfile" 2>&1 &
    local pid=$!
    log "Launched training PID=$pid log=$logfile"
    # Wait for process to actually start
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
        log "Process $pid confirmed alive"
        return 0
    else
        log "ERROR: Process $pid died immediately"
        return 1
    fi
}

check_finished() {
    local logfile
    logfile=$(get_latest_log)
    [ -z "$logfile" ] && return 1
    # Training prints "Entrenamiento completado" or similar when done
    grep -qi "completado\|finished\|done.*epoch\|Saving final" "$logfile" 2>/dev/null
}

log "=== Watchdog started (interval=${INTERVAL}s, max_restarts=$MAX_RESTARTS) ==="

while true; do
    sleep "$INTERVAL"

    if check_finished; then
        log "Training appears FINISHED. Watchdog exiting."
        break
    fi

    if is_training_alive; then
        # Training is running — log status
        logfile=$(get_latest_log)
        last_progress=$(grep '\[progress\]' "$logfile" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
        last_val=$(grep '\[VAL\]' "$logfile" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
        log "OK | $last_progress | $last_val"
    else
        # Training died
        logfile=$(get_latest_log)
        last_line=$(tail -1 "$logfile" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
        log "CRASH detected! Last line: $last_line"

        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
            log "ERROR: Reached max restarts ($MAX_RESTARTS). GPU driver likely contaminated."
            log "ACTION NEEDED: Reboot, force GPU high, then restart watchdog."
            break
        fi

        restarts=$((restarts + 1))
        log "Attempting relaunch $restarts/$MAX_RESTARTS..."
        if launch_training; then
            log "Relaunch successful. Waiting for next check."
        else
            log "Relaunch FAILED. Stopping watchdog."
            break
        fi
    fi
done

log "=== Watchdog exiting (restarts used: $restarts/$MAX_RESTARTS) ==="
