#!/bin/bash
# ============================================================
# AIDEEN Training Launcher (TTY sin compositor)
#
# Uso: desde TTY (Ctrl+Alt+F3), login, luego:
#   bash ~/sustainability/aideen/launch_training_tty.sh
#
# Para volver al escritorio despues:
#   sudo systemctl start display-manager
# ============================================================

set -e

echo "=== AIDEEN Training Launcher ==="
echo ""

# 1. Matar compositor si esta corriendo
if pgrep -x cosmic-comp > /dev/null 2>&1; then
    echo "[1/5] Deteniendo compositor COSMIC..."
    sudo systemctl stop display-manager
    sleep 2
    echo "      Compositor detenido."
else
    echo "[1/5] Compositor no esta corriendo. OK."
fi

# 2. Forzar GPU a max clocks
echo "[2/5] Forzando GPU a velocidad maxima..."
echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level > /dev/null
GPU_CLOCK=$(cat /sys/class/drm/card1/device/pp_dpm_sclk | grep '\*' | awk '{print $2}')
echo "      GPU clock: $GPU_CLOCK"

# 3. Verificar VRAM libre
VRAM_USED=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
VRAM_TOTAL=$(cat /sys/class/drm/card1/device/mem_info_vram_total)
VRAM_FREE_MB=$(( (VRAM_TOTAL - VRAM_USED) / 1048576 ))
echo "[3/5] VRAM libre: ${VRAM_FREE_MB} MB / $((VRAM_TOTAL / 1048576)) MB"

# 4. Matar procesos de entrenamiento previos
if pgrep -f "target/release/train" > /dev/null 2>&1; then
    echo "[4/5] Matando entrenamientos previos..."
    pkill -9 -f "target/release/train" 2>/dev/null || true
    sleep 1
else
    echo "[4/5] No hay entrenamientos previos."
fi

# 5. Lanzar entrenamiento
cd ~/sustainability/aideen
LOGFILE="training_full_$(date +%Y%m%d_%H%M%S).log"

echo "[5/5] Lanzando entrenamiento..."
echo "      Log: $LOGFILE"
echo "      Corpus: corpus_combined.txt"
echo "      Checkpoint: model_large"
echo ""

nohup cargo run --release -p aideen-training \
    --features aideen-training/wgpu \
    --bin train -- \
    --file corpus_combined.txt \
    --resume model_large \
    --epochs 1 \
    --log-every 1 \
    --save-every 1 \
    > "$LOGFILE" 2>&1 &

TRAIN_PID=$!
echo "PID: $TRAIN_PID"
echo ""
echo "Esperando 4 minutos para confirmar estabilidad..."
echo "(Si ves lineas [progress] despues de chunk 10, esta funcionando)"
echo ""

# Monitorear por 4 minutos
for i in $(seq 1 16); do
    sleep 15
    ELAPSED=$((i * 15))

    # Verificar que el proceso sigue vivo
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo ""
        echo "*** PROCESO MURIO despues de ${ELAPSED}s ***"
        echo "Ultimas lineas del log:"
        tail -5 "$LOGFILE"
        exit 1
    fi

    # Mostrar ultima linea de progreso
    LAST=$(grep -E "\[progress\]|\[VAL\]|radv" "$LOGFILE" 2>/dev/null | tail -1)
    echo "  [${ELAPSED}s] $LAST"
done

echo ""
echo "=== Entrenamiento estable por 4 minutos ==="
echo "Monitorear: tail -f ~/sustainability/aideen/$LOGFILE"
echo "Volver al escritorio: sudo systemctl start display-manager"
echo "Duracion estimada: ~88 horas"
