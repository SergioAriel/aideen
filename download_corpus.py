"""Download a training corpus for AIDEEN from HuggingFace.

Strategy: Download FineWeb-Edu (high quality web text) in streaming mode,
extract plain text, combine with existing data, output a single .txt file.

Target: ~50-100MB of clean text (~15-30M tokens with BPE).
This is enough for a meaningful loss curve on a 30M param model.
"""

import os
import sys
from pathlib import Path

# Activate venv if needed
venv_path = Path(__file__).parent / ".venv" / "bin" / "activate_this.py"

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "corpus_fineweb_50m.txt"
TARGET_BYTES = 50 * 1024 * 1024  # 50MB target (~15M tokens)
SEPARATOR = "<|endoftext|>"

def download_fineweb():
    """Download FineWeb-Edu sample in streaming mode."""
    print(f"Downloading FineWeb-Edu (target: {TARGET_BYTES // 1024 // 1024}MB)...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )

    total_bytes = 0
    doc_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue

            f.write(text.strip())
            f.write(f"\n{SEPARATOR}\n")

            total_bytes += len(text.encode("utf-8"))
            doc_count += 1

            if doc_count % 500 == 0:
                mb = total_bytes / 1024 / 1024
                print(f"  {doc_count} docs, {mb:.1f}MB...")

            if total_bytes >= TARGET_BYTES:
                break

    mb = total_bytes / 1024 / 1024
    print(f"Done: {doc_count} docs, {mb:.1f}MB -> {OUTPUT_FILE}")
    return str(OUTPUT_FILE)


def download_wikipedia_simple():
    """Fallback: download Simple Wikipedia (smaller, faster)."""
    print("Downloading Wikipedia (simple) as fallback...")

    ds = load_dataset(
        "wikipedia",
        "20220301.simple",
        split="train",
        streaming=True,
    )

    total_bytes = 0
    doc_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 200:
                continue

            f.write(text.strip())
            f.write(f"\n{SEPARATOR}\n")

            total_bytes += len(text.encode("utf-8"))
            doc_count += 1

            if doc_count % 1000 == 0:
                mb = total_bytes / 1024 / 1024
                print(f"  {doc_count} docs, {mb:.1f}MB...")

            if total_bytes >= TARGET_BYTES:
                break

    mb = total_bytes / 1024 / 1024
    print(f"Done: {doc_count} docs, {mb:.1f}MB -> {OUTPUT_FILE}")
    return str(OUTPUT_FILE)


if __name__ == "__main__":
    try:
        download_fineweb()
    except Exception as e:
        print(f"FineWeb failed ({e}), trying Wikipedia...")
        try:
            download_wikipedia_simple()
        except Exception as e2:
            print(f"Wikipedia also failed: {e2}")
            sys.exit(1)
