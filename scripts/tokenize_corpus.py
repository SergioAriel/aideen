#!/usr/bin/env python3
"""
tokenize_corpus.py -- Tokenize the raw corpus with the AIDEEN BPE 64K tokenizer.

Reads  : data/corpus/raw_corpus.txt
Writes : data/corpus/train.tokens.bin   (90 %)
         data/corpus/val.tokens.bin     (10 %)

Token format: sequence of little-endian u32 token IDs.

Requires: pip install tokenizers
"""

import struct
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("ERROR: 'tokenizers' package not found.")
    print("Install it with:  pip install tokenizers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
TOKENIZER_PATH  = PROJECT_ROOT / "aideen-backbone" / "tokenizer.json"
CORPUS_PATH     = PROJECT_ROOT / "data" / "corpus" / "raw_corpus.txt"
TRAIN_OUT       = PROJECT_ROOT / "data" / "corpus" / "train.tokens.bin"
VAL_OUT         = PROJECT_ROOT / "data" / "corpus" / "val.tokens.bin"

TRAIN_FRACTION  = 0.9  # 90 / 10 split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_u32_le(path: Path, ids: list[int]) -> None:
    """Write a list of token IDs as little-endian u32 binary."""
    with path.open("wb") as f:
        for token_id in ids:
            f.write(struct.pack("<I", token_id))


def format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1_048_576:
        return f"{n_bytes / 1024:.1f} KB"
    else:
        return f"{n_bytes / 1_048_576:.2f} MB"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  AIDEEN Corpus Tokenizer")
    print("=" * 64)
    print()

    # 1. Load tokenizer
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        sys.exit(1)

    print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size:,}")
    print()

    # 2. Read corpus
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        print("Run  python scripts/fetch_corpus.py  first.")
        sys.exit(1)

    print(f"Reading corpus from {CORPUS_PATH} ...")
    text = CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    text_size = len(text.encode("utf-8"))
    print(f"  Corpus size: {format_size(text_size)} ({len(text):,} chars)")
    print()

    # 3. Encode
    print("Tokenizing (this may take a moment) ...")
    encoding = tokenizer.encode(text)
    token_ids: list[int] = encoding.ids
    n_tokens = len(token_ids)
    print(f"  Total tokens: {n_tokens:,}")
    print(f"  Chars / token: {len(text) / n_tokens:.2f}")
    print()

    # 4. Split train / val
    split_idx = int(n_tokens * TRAIN_FRACTION)
    train_ids = token_ids[:split_idx]
    val_ids   = token_ids[split_idx:]

    print(f"Splitting {TRAIN_FRACTION:.0%} / {1 - TRAIN_FRACTION:.0%} ...")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens  : {len(val_ids):,}")
    print()

    # 5. Write binary files
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {TRAIN_OUT} ...")
    write_u32_le(TRAIN_OUT, train_ids)
    train_size = TRAIN_OUT.stat().st_size
    print(f"  {format_size(train_size)}")

    print(f"Writing {VAL_OUT} ...")
    write_u32_le(VAL_OUT, val_ids)
    val_size = VAL_OUT.stat().st_size
    print(f"  {format_size(val_size)}")

    print()
    print("Done.")
    print(f"  train : {len(train_ids):,} tokens  ({format_size(train_size)})")
    print(f"  val   : {len(val_ids):,} tokens  ({format_size(val_size)})")
    print(f"  total : {n_tokens:,} tokens")


if __name__ == "__main__":
    main()
