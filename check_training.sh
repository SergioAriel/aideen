#!/bin/bash
# Quick health check for AIDEEN training.
# Uso: ./check_training.sh

cd "$(dirname "$0")"

PID=$(cat training_logs/training.pid 2>/dev/null)

echo "═══════════════════════════════════════════"
echo "  AIDEEN Training Monitor"
echo "═══════════════════════════════════════════"

# Check if process is alive
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    MEM=$(ps -p "$PID" -o rss= 2>/dev/null | awk '{printf "%.0f MB", $1/1024}')
    CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null)
    echo "  Estado:  RUNNING (PID $PID)"
    echo "  RAM:     $MEM"
    echo "  CPU:     $CPU%"
else
    echo "  Estado:  STOPPED"
    echo "  Para resumir: ./resume_training.sh"
fi

echo ""

# Latest log
LOG=$(ls -t training_logs/run_*.log training_logs/resume_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo "  Log: $LOG"
    echo ""

    # Last val loss
    LAST_VAL=$(grep "VAL" "$LOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
    echo "  Último VAL: $LAST_VAL"

    # Last progress
    LAST_PROG=$(grep "progress" "$LOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
    echo "  Último progress: $LAST_PROG"

    # Stats
    TOTAL_CHUNKS=$(grep -c "progress" "$LOG" 2>/dev/null || echo 0)
    FIRST_VAL=$(grep "val_loss" "$LOG" | head -1 | grep -oP 'val_loss=[\d.]+' | cut -d= -f2)
    BEST_VAL=$(grep "val_loss" "$LOG" | grep -oP 'val_loss=[\d.]+' | cut -d= -f2 | sort -n | head -1)
    LAST_VAL_NUM=$(grep "val_loss" "$LOG" | tail -1 | grep -oP 'val_loss=[\d.]+' | cut -d= -f2)

    echo ""
    echo "  Chunks procesados: ~$((TOTAL_CHUNKS * 10))"
    echo "  Val loss inicial:  $FIRST_VAL"
    echo "  Val loss mejor:    $BEST_VAL"
    echo "  Val loss actual:   $LAST_VAL_NUM"

    # Checkpoint
    if [ -f "model_large.aidn" ]; then
        CKPT_TIME=$(stat -c %y model_large.aidn 2>/dev/null | cut -d. -f1)
        CKPT_SIZE=$(du -sh model_large.aidn | cut -f1)
        echo ""
        echo "  Checkpoint: model_large.aidn ($CKPT_SIZE)"
        echo "  Guardado:   $CKPT_TIME"
    fi
fi

echo ""
echo "═══════════════════════════════════════════"
