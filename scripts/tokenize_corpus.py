#!/usr/bin/env python3
"""
tokenize_corpus.py -- Tokenize the raw corpus with the AIDEEN BPE 64K tokenizer.

Reads  : data/corpus/raw_corpus.txt
Writes : data/corpus/train.tokens.bin   (90 %)
         data/corpus/val.tokens.bin     (10 %)

Token format: sequence of little-endian u32 token IDs.

Processes the corpus in streaming chunks to avoid OOM on large files.

Requires: pip install tokenizers
"""

import array
import os
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
CHUNK_LINES     = 500_000  # lines per chunk (~50-100 MB depending on line length)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1_048_576:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1_073_741_824:
        return f"{n_bytes / 1_048_576:.2f} MB"
    else:
        return f"{n_bytes / 1_073_741_824:.2f} GB"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  AIDEEN Corpus Tokenizer (streaming)")
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

    # 2. Check corpus
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        print("Run  python scripts/fetch_corpus.py  first.")
        sys.exit(1)

    corpus_bytes = CORPUS_PATH.stat().st_size
    print(f"Corpus: {CORPUS_PATH}")
    print(f"  Size on disk: {format_size(corpus_bytes)}")
    print(f"  Chunk size: {CHUNK_LINES:,} lines")
    print()

    # 3. Tokenize in streaming chunks, write to a single temporary bin file
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = TRAIN_OUT.with_suffix(".tmp")

    n_tokens = 0
    n_chars = 0
    n_chunks = 0
    buf = array.array("I")  # unsigned 32-bit, native (we'll write LE below)

    print("Tokenizing ...")
    with open(CORPUS_PATH, "r", encoding="utf-8", errors="replace") as fin, \
         open(tmp_out, "wb") as fout:
        lines: list[str] = []
        for line in fin:
            lines.append(line)
            if len(lines) >= CHUNK_LINES:
                chunk_text = "".join(lines)
                n_chars += len(chunk_text)
                encoding = tokenizer.encode(chunk_text)
                ids = encoding.ids
                n_tokens += len(ids)
                n_chunks += 1
                # Write as little-endian u32
                fout.write(struct.pack(f"<{len(ids)}I", *ids))
                lines.clear()
                print(f"  chunk {n_chunks}: +{len(ids):,} tokens  "
                      f"(total: {n_tokens:,}, mem ~{format_size(buf.buffer_info()[1] * buf.itemsize)})")

        # Flush remaining lines
        if lines:
            chunk_text = "".join(lines)
            n_chars += len(chunk_text)
            encoding = tokenizer.encode(chunk_text)
            ids = encoding.ids
            n_tokens += len(ids)
            n_chunks += 1
            fout.write(struct.pack(f"<{len(ids)}I", *ids))
            print(f"  chunk {n_chunks} (final): +{len(ids):,} tokens  (total: {n_tokens:,})")

    print()
    print(f"  Total tokens: {n_tokens:,}")
    print(f"  Chars / token: {n_chars / n_tokens:.2f}" if n_tokens else "  (no tokens)")
    print()

    # 4. Split train / val by reading the temp file
    split_token = int(n_tokens * TRAIN_FRACTION)
    total_bytes = tmp_out.stat().st_size
    split_byte = split_token * 4  # each token is 4 bytes (u32)

    print(f"Splitting {TRAIN_FRACTION:.0%} / {1 - TRAIN_FRACTION:.0%} ...")
    print(f"  Train tokens: {split_token:,}")
    print(f"  Val tokens  : {n_tokens - split_token:,}")
    print()

    # Stream-copy from tmp to train and val files
    COPY_BUF = 64 * 1024 * 1024  # 64 MB copy buffer

    with open(tmp_out, "rb") as src:
        with open(TRAIN_OUT, "wb") as dst:
            remaining = split_byte
            while remaining > 0:
                chunk = src.read(min(COPY_BUF, remaining))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)

        with open(VAL_OUT, "wb") as dst:
            while True:
                chunk = src.read(COPY_BUF)
                if not chunk:
                    break
                dst.write(chunk)

    tmp_out.unlink()

    train_size = TRAIN_OUT.stat().st_size
    val_size = VAL_OUT.stat().st_size

    print("Done.")
    print(f"  train : {split_token:,} tokens  ({format_size(train_size)})")
    print(f"  val   : {n_tokens - split_token:,} tokens  ({format_size(val_size)})")
    print(f"  total : {n_tokens:,} tokens")


if __name__ == "__main__":
    main()
