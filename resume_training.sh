#!/bin/bash
# Resume AIDEEN training from the latest checkpoint.
#
# Uso:
#   ./resume_training.sh          # resume con los mismos params
#   ./resume_training.sh 3        # resume con 3 epochs restantes
#
# El checkpoint se guarda cada 500 chunks en model_large.aidn + model_large.opt
# Si el proceso muere, este script retoma exactamente donde quedó.

set -e
cd "$(dirname "$0")"

EPOCHS=${1:-5}
LOGFILE="training_logs/resume_$(date +%Y%m%d_%H%M%S).log"

# Verificar que existe un checkpoint
if [ ! -f "model_large.aidn" ]; then
    echo "ERROR: No se encontró model_large.aidn"
    echo "No hay checkpoint desde el cual resumir."
    echo "Para empezar de cero: cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus_combined.txt --epochs 5"
    exit 1
fi

CKPT_SIZE=$(du -sh model_large.aidn | cut -f1)
echo "Checkpoint encontrado: model_large.aidn ($CKPT_SIZE)"
echo "Resumiendo training con $EPOCHS epochs..."
echo "Log: $LOGFILE"

nohup cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file corpus_combined.txt \
    --resume model_large \
    --epochs "$EPOCHS" \
    --log-every 5 \
    --save-every 500 \
    > "$LOGFILE" 2>&1 &

NEWPID=$!
echo "$NEWPID" > training_logs/training.pid
echo "PID: $NEWPID"
echo ""
echo "Monitorear: tail -f $LOGFILE"
echo "Último progreso: grep 'VAL\|progress' $LOGFILE | tail -10"
