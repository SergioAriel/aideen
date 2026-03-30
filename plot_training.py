#!/usr/bin/env python3
"""Plot AIDEEN training loss curves from log files.

Usage:
    python plot_training.py                    # auto-detect latest log
    python plot_training.py training_logs/run_XXXX.log

Generates:
    docs/figures/loss_curve.png
    docs/figures/throughput.png
    docs/figures/training_report.png
    docs/figures/loss_curve_overview.png
    docs/figures/convergence_table.txt
"""

import re
import sys
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    train_chunks, train_losses, train_tps, train_times = [], [], [], []
    val_chunks, val_losses = [], []

    with open(path, "r", errors="replace") as f:
        for line in f:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = re.search(
                r"\[progress\]\s+chunk\s+(\d+)\s+loss=([\d.]+)\s+tps=\s*([\d.]+)\s+time=([\d.]+)s",
                clean,
            )
            if m:
                train_chunks.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                train_tps.append(float(m.group(3)))
                train_times.append(float(m.group(4)))
                continue
            m = re.search(r"\[VAL\]\s+chunk\s+(\d+)\s+val_loss=([\d.]+)", clean)
            if m:
                val_chunks.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))

    return {
        "train_chunks": train_chunks,
        "train_losses": train_losses,
        "train_tps": train_tps,
        "train_times": train_times,
        "val_chunks": val_chunks,
        "val_losses": val_losses,
    }


def smooth(vals, w):
    return [sum(vals[max(0, i - w) : i + 1]) / min(i + 1, w) for i in range(len(vals))]


def find_latest_log():
    logs = sorted(glob("training_logs/run_*.log") + glob("training_logs/resume_*.log"))
    if not logs:
        print("No log files found in training_logs/")
        sys.exit(1)
    return logs[-1]


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    print(f"Parsing: {log_path}")

    data = parse_log(log_path)
    tc, tl = data["train_chunks"], data["train_losses"]
    vc, vl = data["val_chunks"], data["val_losses"]
    tps = data["train_tps"]
    tt = data["train_times"]

    print(f"  {len(tl)} train points, {len(vl)} val points")
    if not tl and not vl:
        print("No data yet.")
        return

    fig_dir = Path("docs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Filter val outliers
    fvc = [c for c, l in zip(vc, vl) if 0.5 < l < 8]
    fvl = [l for l in vl if 0.5 < l < 8]

    w = max(1, len(tl) // 30)
    wv = max(1, len(fvl) // 20) if fvl else 1
    wt = max(1, len(tps) // 15) if tps else 1
    avg_tps = sum(tps) / len(tps) if tps else 0
    hours = tt[-1] / 3600 if tt else 0

    # Detect epoch boundaries (assumes ~17203 chunks per epoch, but auto-detect)
    # Look for chunk number resets or big jumps
    chunks_per_epoch = None
    if len(tc) > 100:
        for i in range(1, len(tc)):
            if tc[i] < tc[i - 1]:
                chunks_per_epoch = tc[i - 1]
                break

    # ── REPORT (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AIDEEN Training Report\n"
        f"DEQ+Mamba, D_R=512, K=4 slots | {len(tl) * 10:,} steps",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Loss curve
    ax = axes[0][0]
    ax.plot(tc, tl, color="#2563eb", alpha=0.15, linewidth=0.5)
    if w > 1:
        ax.plot(tc[w - 1 :], smooth(tl, w)[w - 1 :], color="#2563eb", linewidth=2.2, label=f"Train (avg {w})")
    if fvl:
        ax.scatter(fvc, fvl, color="#dc2626", s=4, alpha=0.15, zorder=1)
        ax.plot(fvc[wv - 1 :], smooth(fvl, wv)[wv - 1 :], color="#dc2626", linewidth=2.2, label=f"Val (avg {wv})")
    ax.axhline(y=10.83, color="gray", linestyle="--", alpha=0.3)
    ax.text(max(tc) * 0.5, 10.4, "Random baseline (10.83)", fontsize=8, color="gray")
    ax.set_xlabel("Steps (chunks of 256 tokens)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(bottom=0)

    # Throughput
    ax = axes[0][1]
    ax.plot(tc, tps, color="#059669", alpha=0.3, linewidth=0.5)
    if wt > 1:
        ax.plot(tc[wt - 1 :], smooth(tps, wt)[wt - 1 :], color="#059669", linewidth=2.5)
    ax.axhline(y=avg_tps, color="#059669", linestyle="--", alpha=0.4)
    ax.text(max(tc) * 0.6, avg_tps + 1, f"Avg: {avg_tps:.1f} tps", fontsize=10, color="#059669", fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Tokens/second")
    ax.set_title("Throughput (AMD Radeon 780M iGPU)")
    ax.grid(True, alpha=0.15)

    # Val loss zoomed
    ax = axes[1][0]
    if fvl:
        ax.scatter(fvc, fvl, color="#dc2626", s=5, alpha=0.2)
        wv2 = max(1, len(fvl) // 12)
        sm_val = smooth(fvl, wv2)
        ax.plot(fvc[wv2 - 1 :], sm_val[wv2 - 1 :], color="#dc2626", linewidth=2.5, label=f"Val trend (avg {wv2})")
        best_val = min(fvl)
        best_idx = fvl.index(best_val)
        ax.annotate(
            f"Best: {best_val:.3f}",
            xy=(fvc[best_idx], best_val),
            xytext=(fvc[best_idx] + max(tc) * 0.05, best_val + 0.4),
            arrowprops=dict(arrowstyle="->", color="#dc2626"),
            fontsize=9, color="#dc2626", fontweight="bold",
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation Loss (filtered)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Summary
    ax = axes[1][1]
    ax.axis("off")
    rows = [
        ("Steps (chunks)", f"{max(tc):,}"),
        ("Tokens processed", f"{max(tc) * 256:,}"),
        ("", ""),
        ("Train loss (start)", f"{tl[0]:.4f}"),
        ("Train loss (now)", f"{tl[-1]:.4f}"),
        ("Val loss (start)", f"{vl[0]:.4f}" if vl else "—"),
        ("Val loss (best)", f"{min(fvl):.4f}" if fvl else "—"),
        ("Val loss (now)", f"{vl[-1]:.4f}" if vl else "—"),
        ("", ""),
        ("Throughput", f"{avg_tps:.1f} tokens/sec"),
        ("Wall time", f"{hours:.1f} hours"),
        ("GPU", "Radeon 780M (iGPU, Vulkan)"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.95 - i * 0.07
        if not k and not v:
            continue
        bold = "bold" if k in ("Train loss (now)", "Val loss (best)", "Throughput", "Steps (chunks)") else "normal"
        ax.text(0.05, y, k, fontsize=10.5, fontweight=bold, transform=ax.transAxes, va="top", fontfamily="monospace")
        ax.text(0.55, y, v, fontsize=10.5, fontweight=bold, transform=ax.transAxes, va="top", fontfamily="monospace")
    ax.set_title("Summary", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "training_report.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: training_report.png")

    # ── Grant curve (clean) ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    if w > 1:
        ax2.plot(tc[w - 1 :], smooth(tl, w)[w - 1 :], color="#2563eb", linewidth=2.5, label="Train loss", zorder=3)
    if fvl and wv > 1:
        ax2.plot(fvc[wv - 1 :], smooth(fvl, wv)[wv - 1 :], color="#dc2626", linewidth=2.5, label="Validation loss", zorder=3)
        ax2.scatter(fvc, fvl, color="#dc2626", s=4, alpha=0.12, zorder=2)
    ax2.axhline(y=10.83, color="gray", linestyle="--", alpha=0.25)
    ax2.text(max(tc) * 0.02, 10.3, "Random init (ln 50257 ≈ 10.83)", fontsize=9, color="gray")
    ax2.set_xlabel("Training steps (chunks of 256 tokens)", fontsize=12)
    ax2.set_ylabel("Cross-entropy loss", fontsize=12)
    ax2.set_title(
        "AIDEEN: DEQ+SSM Training Convergence\n"
        "30M parameters, consumer iGPU (AMD Radeon 780M, Vulkan)",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.12)
    ax2.set_ylim(bottom=0, top=12)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "loss_curve_overview.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Saved: loss_curve_overview.png")

    # ── Detail loss ──
    fig3, ax3 = plt.subplots(figsize=(10, 5.5))
    ax3.plot(tc, tl, color="#2563eb", alpha=0.2, linewidth=0.5)
    if w > 1:
        ax3.plot(tc[w - 1 :], smooth(tl, w)[w - 1 :], color="#2563eb", linewidth=2, label=f"Train (avg {w})")
    if fvl:
        ax3.plot(fvc, fvl, color="#dc2626", alpha=0.4, linewidth=0.8, marker=".", markersize=2)
        ax3.plot(fvc[wv - 1 :], smooth(fvl, wv)[wv - 1 :], color="#dc2626", linewidth=2, label=f"Val (avg {wv})")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Loss")
    ax3.set_title("AIDEEN Loss Curve (detail)")
    ax3.legend()
    ax3.grid(True, alpha=0.15)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "loss_curve.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"  Saved: loss_curve.png")

    # ── Throughput ──
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(tc, tps, color="#059669", alpha=0.3, linewidth=0.5)
    if wt > 1:
        ax4.plot(tc[wt - 1 :], smooth(tps, wt)[wt - 1 :], color="#059669", linewidth=2.5)
    ax4.axhline(y=avg_tps, color="#059669", linestyle="--", alpha=0.4)
    ax4.text(max(tc) * 0.7, avg_tps + 1, f"Avg: {avg_tps:.1f} tps", fontsize=10, color="#059669")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Tokens/sec")
    ax4.set_title("Throughput — AMD Radeon 780M (iGPU)")
    ax4.grid(True, alpha=0.15)
    fig4.tight_layout()
    fig4.savefig(fig_dir / "throughput.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print(f"  Saved: throughput.png")

    # ── Summary text ──
    with open(fig_dir / "convergence_table.txt", "w") as f:
        f.write(f"AIDEEN Training Summary\n{'=' * 50}\n")
        f.write(f"Steps:             {max(tc):,}\n")
        f.write(f"Tokens processed:  {max(tc) * 256:,}\n")
        f.write(f"Train loss:        {tl[0]:.4f} -> {tl[-1]:.4f}\n")
        f.write(f"Val loss:          {vl[0]:.4f} -> {vl[-1]:.4f}\n" if vl else "")
        if fvl:
            f.write(f"Val loss (best):   {min(fvl):.4f}\n")
        f.write(f"Throughput:        {avg_tps:.1f} tps\n")
        f.write(f"Wall time:         {hours:.1f}h\n")
        f.write(f"Hardware:          Ryzen 9 8945HS + Radeon 780M (Vulkan)\n")
    print(f"  Saved: convergence_table.txt")


if __name__ == "__main__":
    main()
