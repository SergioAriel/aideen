"""Prepare the combined training corpus for AIDEEN grant deadline training.

Combines:
1. dataset_genesis.txt (2.7MB) - Rust Book + arXiv ML papers
2. train_aideen.txt (2.1MB) - SmolTalk + Rust docs
3. corpus_fineweb_50m.txt (first 10MB) - FineWeb-Edu clean web text

Output: corpus_combined.txt (~15MB, ~4.5M tokens)
At ~53 tps: 1 epoch ≈ 6h, 5 epochs ≈ 30h
"""

from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "corpus_combined.txt"
SEP = "<|endoftext|>"

def read_file(path, max_bytes=None):
    """Read file, optionally truncating at max_bytes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_bytes and len(text.encode("utf-8")) > max_bytes:
        # Truncate at last newline before max_bytes
        encoded = text.encode("utf-8")[:max_bytes]
        text = encoded.decode("utf-8", errors="ignore")
        last_nl = text.rfind("\n")
        if last_nl > 0:
            text = text[:last_nl]
    return text

def main():
    parts = []
    total_bytes = 0

    # 1. Genesis dataset (full)
    genesis = ROOT / "aideen-backbone" / "dataset_genesis.txt"
    if genesis.exists():
        text = read_file(genesis)
        parts.append(text)
        b = len(text.encode("utf-8"))
        total_bytes += b
        print(f"  genesis: {b/1024/1024:.1f}MB")

    # 2. Train aideen (full)
    train = ROOT / "train_aideen.txt"
    if train.exists():
        text = read_file(train)
        parts.append(text)
        b = len(text.encode("utf-8"))
        total_bytes += b
        print(f"  train_aideen: {b/1024/1024:.1f}MB")

    # 3. FineWeb subset (first 10MB)
    fineweb = ROOT / "corpus_fineweb_50m.txt"
    if fineweb.exists():
        text = read_file(fineweb, max_bytes=10 * 1024 * 1024)
        parts.append(text)
        b = len(text.encode("utf-8"))
        total_bytes += b
        print(f"  fineweb_10MB: {b/1024/1024:.1f}MB")

    # Combine with separator
    combined = f"\n{SEP}\n".join(parts)

    OUTPUT.write_text(combined, encoding="utf-8")
    mb = len(combined.encode("utf-8")) / 1024 / 1024
    print(f"\nTotal: {mb:.1f}MB -> {OUTPUT}")
    print(f"Estimated tokens (BPE): ~{int(mb * 300_000):,}")

if __name__ == "__main__":
    main()
