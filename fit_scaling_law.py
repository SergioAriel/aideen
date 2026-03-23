#!/usr/bin/env python3
"""
Ajusta una power law L = a * N^(-alpha) + L_inf a partir de los resultados
del scaling_experiment.sh.

Uso:
    python3 fit_scaling_law.py scaling_results.log

O sin archivo (pega los datos a mano abajo en MANUAL_DATA).
"""

import sys
import re
import math

# ─── Parámetros del modelo ───────────────────────────────────────────────────
# N_params ∝ d_r² × h_slots × n_matrices
# Matrices principales: W_q, W_k, W_v (h×d²) + W_o (d²) + W_hist, W_delta, ...
# Aproximación conservadora: 4 bloques h×d² + varios d²
H_SLOTS = 8

def param_count(d_r):
    """Cuenta de parámetros aproximada (dominada por matrices d²)."""
    # Atención: W_q + W_k + W_v = 3 × h × d²  +  W_o = d²
    attn = (3 * H_SLOTS + 1) * d_r * d_r
    # SSM: W_delta (h×d²) + W_hist (h×h×d?) + A_log (h×d) ≈ h×d²
    ssm = H_SLOTS * d_r * d_r
    # Forget gate: h×d (tiny)
    forget = H_SLOTS * d_r
    # Embeddings: vocab × d_r  (vocab≈200 para tinyshakespeare)
    emb = 200 * d_r
    return attn + ssm + forget + emb

# ─── Parseo del log ──────────────────────────────────────────────────────────
def parse_log(path):
    """Extrae (d_r, loss_final) del log del experimento."""
    results = {}
    current_dr = None
    with open(path) as f:
        for line in f:
            m = re.search(r"d_r=(\d+)", line)
            if m:
                current_dr = int(m.group(1))
            m = re.search(r"Iter\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)", line)
            if m and current_dr is not None:
                it = int(m.group(1))
                loss = float(m.group(2))
                if current_dr not in results or it > results[current_dr][0]:
                    results[current_dr] = (it, loss)
    return {dr: loss for dr, (_, loss) in results.items()}

# ─── Datos manuales (fallback si no hay log) ─────────────────────────────────
MANUAL_DATA = {
    # d_r: loss_final   ← rellená con tus resultados
    # 64:  5.20,
    # 128: 4.85,
    # 256: 4.55,
    # 512: 4.30,
}

# ─── Power law fit ───────────────────────────────────────────────────────────
def fit_power_law(ns, ls):
    """
    Ajusta L = a * N^(-alpha) usando mínimos cuadrados en log-log.
    Devuelve (a, alpha).
    """
    log_n = [math.log(n) for n in ns]
    log_l = [math.log(l) for l in ls]
    n = len(log_n)
    mean_x = sum(log_n) / n
    mean_y = sum(log_l) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_l))
    ss_xx = sum((x - mean_x) ** 2 for x in log_n)
    alpha = -ss_xy / ss_xx   # pendiente en log-log (negativa)
    log_a = mean_y + alpha * mean_x
    a = math.exp(log_a)
    return a, alpha

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        data = parse_log(sys.argv[1])
    else:
        data = MANUAL_DATA

    if not data:
        print("No hay datos. Ejecuta ./scaling_experiment.sh primero.")
        return

    print("\n=== SCALING LAW RESULTS ===\n")
    print(f"{'d_r':>6}  {'N_params':>10}  {'Loss':>8}  {'log(N)':>8}  {'log(L)':>8}")
    print("-" * 52)

    dr_vals = sorted(data.keys())
    ns = []
    ls = []
    for dr in dr_vals:
        loss = data[dr]
        n = param_count(dr)
        ns.append(n)
        ls.append(loss)
        print(f"{dr:>6}  {n:>10,}  {loss:>8.4f}  {math.log(n):>8.3f}  {math.log(loss):>8.3f}")

    if len(ns) >= 2:
        a, alpha = fit_power_law(ns, ls)
        print(f"\nPower law fit: L = {a:.4f} × N^(-{alpha:.4f})")
        print(f"  alpha = {alpha:.4f}  (Chinchilla Transformer: ~0.076, GPT-3: ~0.05-0.08)")
        if alpha > 0.08:
            print("  ✓ Mejor que Transformer estándar (alpha más alto = escala mejor)")
        elif alpha > 0.04:
            print("  ~ Similar a Transformer estándar")
        else:
            print("  ✗ Escala peor que Transformer (revisar LR o arquitectura)")

        # Extrapolación a d_r=1024, 2048
        print("\nExtrapolación:")
        for dr_pred in [1024, 2048, 4096]:
            n_pred = param_count(dr_pred)
            l_pred = a * (n_pred ** -alpha)
            print(f"  d_r={dr_pred:>4}: N≈{n_pred:>12,}  L_pred≈{l_pred:.4f}")

if __name__ == "__main__":
    main()
