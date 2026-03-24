"""Build a proper validation set from FineWeb-Edu.

Downloads documents that were NOT used in the training corpus.
The training corpus used the first ~12,500 docs. We skip those
and take the next batch as validation.

Output: val_fineweb.txt (~5MB, ~1.5M tokens)
"""

from pathlib import Path
from datasets import load_dataset

OUTPUT = Path(__file__).parent / "val_fineweb.txt"
TRAIN_DOCS_USED = 12_500  # docs already in corpus_fineweb_50m.txt
TARGET_BYTES = 5 * 1024 * 1024  # 5MB — enough for reliable val_loss
SEPARATOR = "<|endoftext|>"


def main():
    print(f"Downloading FineWeb-Edu validation set...")
    print(f"  Skipping first {TRAIN_DOCS_USED} docs (used in training)")
    print(f"  Target: {TARGET_BYTES // 1024 // 1024}MB")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )

    total_bytes = 0
    doc_count = 0
    skipped = 0

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for example in ds:
            # Skip docs that are in the training set
            skipped += 1
            if skipped <= TRAIN_DOCS_USED:
                if skipped % 2000 == 0:
                    print(f"  Skipping... {skipped}/{TRAIN_DOCS_USED}")
                continue

            text = example.get("text", "")
            if not text or len(text) < 100:
                continue

            f.write(text.strip())
            f.write(f"\n{SEPARATOR}\n")

            total_bytes += len(text.encode("utf-8"))
            doc_count += 1

            if doc_count % 500 == 0:
                mb = total_bytes / 1024 / 1024
                print(f"  {doc_count} val docs, {mb:.1f}MB...")

            if total_bytes >= TARGET_BYTES:
                break

    mb = total_bytes / 1024 / 1024
    print(f"\nDone: {doc_count} docs, {mb:.1f}MB -> {OUTPUT}")
    print(f"  (skipped {TRAIN_DOCS_USED} training docs)")
    print(f"  Estimated tokens: ~{int(mb * 300_000):,}")


if __name__ == "__main__":
    main()
