#!/bin/bash
# Download and prepare the AIDEEN training corpus.
# Produces corpus_combined.txt (~15MB, ~3.76M BPE tokens)
#
# Sources:
#   1. The Rust Programming Language book (doc.rust-lang.org)
#   2. SmolTalk dataset (HuggingFace)
#   3. FineWeb-Edu clean web text (HuggingFace, first 10MB)
#
# Usage: bash download_corpus.sh

set -e
cd "$(dirname "$0")"

echo "=== AIDEEN corpus preparation ==="

# ── 1. Rust Book ──────────────────────────────────────────────
RUST_BOOK="rust_book.txt"
if [ ! -f "$RUST_BOOK" ]; then
    echo "[1/3] Downloading The Rust Programming Language..."
    # Download the mdBook source and extract text
    TMP_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/rust-lang/book.git "$TMP_DIR/book" 2>/dev/null
    find "$TMP_DIR/book/src" -name "*.md" -exec cat {} + > "$RUST_BOOK"
    rm -rf "$TMP_DIR"
    echo "  $(wc -c < "$RUST_BOOK" | xargs) bytes"
else
    echo "[1/3] Rust Book already downloaded ($(wc -c < "$RUST_BOOK" | xargs) bytes)"
fi

# ── 2. SmolTalk ───────────────────────────────────────────────
SMOLTALK="smoltalk.txt"
if [ ! -f "$SMOLTALK" ]; then
    echo "[2/3] Downloading SmolTalk dataset..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train', streaming=True)
with open('$SMOLTALK', 'w') as f:
    for i, row in enumerate(ds):
        if i >= 5000: break
        for msg in row.get('messages', []):
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            f.write(f'{role}: {content}\n')
        f.write('\n')
print(f'  SmolTalk: {i+1} conversations')
" 2>/dev/null || {
        echo "  [fallback] datasets library not available, skipping SmolTalk"
        echo "  Install with: pip3 install datasets"
        touch "$SMOLTALK"
    }
    echo "  $(wc -c < "$SMOLTALK" | xargs) bytes"
else
    echo "[2/3] SmolTalk already downloaded ($(wc -c < "$SMOLTALK" | xargs) bytes)"
fi

# ── 3. FineWeb-Edu ────────────────────────────────────────────
FINEWEB="fineweb_10m.txt"
if [ ! -f "$FINEWEB" ]; then
    echo "[3/3] Downloading FineWeb-Edu (first 10MB)..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', streaming=True)
total = 0
limit = 10 * 1024 * 1024  # 10MB
with open('$FINEWEB', 'w') as f:
    for row in ds:
        text = row.get('text', '')
        f.write(text + '\n\n')
        total += len(text.encode('utf-8'))
        if total >= limit:
            break
print(f'  FineWeb: {total/1024/1024:.1f}MB')
" 2>/dev/null || {
        echo "  [fallback] datasets library not available, skipping FineWeb"
        echo "  Install with: pip3 install datasets"
        touch "$FINEWEB"
    }
    echo "  $(wc -c < "$FINEWEB" | xargs) bytes"
else
    echo "[3/3] FineWeb already downloaded ($(wc -c < "$FINEWEB" | xargs) bytes)"
fi

# ── Combine ───────────────────────────────────────────────────
OUTPUT="corpus_combined.txt"
echo ""
echo "Combining into $OUTPUT..."
SEP="<|endoftext|>"
cat "$RUST_BOOK" > "$OUTPUT"
echo -e "\n${SEP}\n" >> "$OUTPUT"
cat "$SMOLTALK" >> "$OUTPUT"
echo -e "\n${SEP}\n" >> "$OUTPUT"
cat "$FINEWEB" >> "$OUTPUT"

SIZE=$(wc -c < "$OUTPUT" | xargs)
echo "Done: $OUTPUT (${SIZE} bytes, ~$(echo "$SIZE / 1048576" | bc)MB)"
echo ""
echo "To train: cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus_combined.txt --epochs 1"
