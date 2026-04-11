#!/usr/bin/env python3
"""
build_benchmark_corpus.py — Corpus de benchmark para evaluar dependencias largas en AIDEEN.

Estándar de la industria (2024-2025):
  - Texto real, no sintético (PG-19: libros completos de Project Gutenberg)
  - Documentos enteros o chunks continuos del mismo documento por batch
  - Métrica principal: perplexity en held-out text
  - Evaluación separada por longitud de contexto

Por qué PG-19:
  - Es el benchmark estándar para long-range language modeling
  - Libros completos con dependencias largas naturales (personajes, tramas, conceptos)
  - Sin contaminación de datos modernos
  - Usado en los papers de Transformer-XL, Compressive Transformer, Mamba, etc.

Diseño de batches:
  - Cada batch contiene chunks continuos del MISMO libro
  - Un libro = un tema cohesivo = dependencias intra-batch reales
  - Si el libro es más corto que el batch, se rellena con el siguiente libro del mismo autor
  - Si el libro es más largo, se parten en batches consecutivos (mismo libro)

Métricas a reportar:
  1. Perplexity en val set (principal)
  2. Bits per byte (BPB) — normalizado por tokenización
  3. Comparación AIDEEN vs transformer baseline del mismo tamaño

Uso:
  pip install datasets huggingface_hub
  python3 build_benchmark_corpus.py \\
    --out-dir /Users/sergiosolis/Programacion/AIDEEN \\
    --ctx-len 512 --batch-size 4 \\
    --n-train-books 50 --n-val-books 10

Requisitos:
  pip install datasets huggingface_hub tqdm
"""

import argparse
import os
import sys
import random
import math
from pathlib import Path

DOC_MARKER = "<|endoftext|>"

# ---------------------------------------------------------------------------
# Dependencias opcionales — reportar error claro si faltan
# ---------------------------------------------------------------------------

def check_deps():
    missing = []
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    if missing:
        print(f"ERROR: Faltan dependencias: {', '.join(missing)}")
        print(f"Instalar con: pip install {' '.join(missing)}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Carga de PG-19
# ---------------------------------------------------------------------------

def load_pg19(split: str, n_books: int, seed: int):
    """
    Carga n_books libros del dataset PG-19.
    split: "train" o "validation" o "test"
    Retorna lista de (book_id, title, text).
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"  Cargando PG-19 split='{split}'...")
    # El dataset histórico `deepmind/pg19` quedó como loading script legacy.
    # Probamos primero mirrors en formato estándar compatibles con `datasets`
    # moderno y streaming.
    dataset_candidates = [
        "emozilla/pg19",
        "deepmind/pg19",
    ]
    ds = None
    last_error = None
    for dataset_name in dataset_candidates:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            print(f"    Fuente activa: {dataset_name}")
            break
        except Exception as e:
            last_error = e
            print(f"    WARN: no se pudo abrir {dataset_name}: {e}")
    if ds is None:
        raise RuntimeError(f"No se pudo cargar PG-19 desde ninguna fuente: {last_error}")

    rng = random.Random(seed)
    books = []
    print(f"  Leyendo hasta {n_books} libros...")

    # Con streaming tomamos los primeros n_books después de shuffle
    # Para val/test son pocos libros así que los tomamos todos
    for i, item in enumerate(ds):
        if len(books) >= n_books:
            break
        text = item.get("text", "")
        if len(text) < 5000:  # skip libros muy cortos
            continue
        book_id = item.get("short_book_title", f"book_{i}")
        books.append((book_id, text))

    print(f"  Cargados {len(books)} libros")
    return books


# ---------------------------------------------------------------------------
# Construcción de corpus por batches cohesivos
# ---------------------------------------------------------------------------

def build_corpus_from_books(
    books: list,
    ctx_len: int,
    batch_size: int,
    rng: random.Random,
    max_batches: int = None,
) -> str:
    """
    Convierte una lista de libros en un corpus donde la continuidad
    intra-libro se preserva y las fronteras de documento quedan
    marcadas explícitamente.

    Estructura de output:
      - Libros/chunks continuos del mismo libro
      - Fronteras de documento marcadas con DOC_MARKER
      - Sin mezclar chunks de libros distintos dentro del mismo stream

    Por qué texto continuo del mismo libro:
      - Las dependencias token-to-token dentro del batch son reales
      - El modelo puede aprender a usar la memoria para resolver
        referencias a personajes/conceptos mencionados 200 tokens antes
      - Mezclando libros, las dependencias serían ruido

    Nota sobre tokens vs caracteres:
      BPE tokenization ~= 4 chars/token para inglés.
      Usamos chars como proxy — el tokenizador real se aplica en training.
    """
    chars_per_batch = ctx_len * batch_size * 4  # 4 chars/token aprox
    docs = []
    total_batches = 0

    # El orden de libros puede barajarse para train, pero la continuidad
    # dentro de cada libro no debe romperse.
    ordered_books = books[:]
    rng.shuffle(ordered_books)
    per_book_batch_cap = None
    if max_batches and ordered_books:
        per_book_batch_cap = max(1, math.ceil(max_batches / len(ordered_books)))

    for book_id, text in ordered_books:
        if max_batches and total_batches >= max_batches:
            break

        # Normalizar texto: eliminar exceso de espacios/newlines
        # pero preservar estructura de párrafos
        import re
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        # Preservar continuidad dentro del libro: concatenamos sus chunks
        # consecutivos como un solo documento del stream.
        pos = 0
        book_parts = []
        book_batches = 0
        while pos < len(text):
            if max_batches and total_batches >= max_batches:
                break
            if per_book_batch_cap is not None and book_batches >= per_book_batch_cap:
                break
            chunk = text[pos: pos + chars_per_batch]
            if len(chunk) < chars_per_batch // 4:
                break
            book_parts.append(chunk)
            pos += chars_per_batch
            total_batches += 1
            book_batches += 1

        if book_parts:
            docs.append("\n".join(book_parts).strip())
            print(f"    '{book_id[:50]}': {book_batches} batches")

    print(f"  Total documentos: {len(docs)}")
    print(f"  Total chunks/batches: {total_batches}")
    separator = f"\n{DOC_MARKER}\n"
    return separator.join(docs).strip() + "\n"


# ---------------------------------------------------------------------------
# Fallback: texto libre de Project Gutenberg (sin HuggingFace)
# ---------------------------------------------------------------------------

GUTENBERG_URLS = [
    # Dominio público, dependencias largas, bien conocidos
    ("Pride and Prejudice",        "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("Moby Dick",                  "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("War and Peace",              "https://www.gutenberg.org/files/2600/2600-0.txt"),
    ("The Count of Monte Cristo",  "https://www.gutenberg.org/files/1184/1184-0.txt"),
    ("Middlemarch",                "https://www.gutenberg.org/files/145/145-0.txt"),
    ("David Copperfield",          "https://www.gutenberg.org/files/766/766-0.txt"),
    ("Anna Karenina",              "https://www.gutenberg.org/files/1399/1399-0.txt"),
    ("Don Quixote",                "https://www.gutenberg.org/files/996/996-0.txt"),
    ("Ulysses",                    "https://www.gutenberg.org/files/4300/4300-0.txt"),
    ("Great Expectations",         "https://www.gutenberg.org/files/1400/1400-0.txt"),
]


def load_from_gutenberg(n_books: int, seed: int):
    """Fallback: descarga directa de Project Gutenberg sin HuggingFace."""
    import urllib.request
    import re

    rng = random.Random(seed)
    selection = GUTENBERG_URLS[:n_books]
    books = []

    for title, url in selection:
        print(f"  Descargando: {title}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            # Eliminar header/footer de Gutenberg
            start = raw.find("*** START OF")
            end   = raw.find("*** END OF")
            if start != -1:
                raw = raw[start + 100:]
            if end != -1:
                raw = raw[:end]
            books.append((title, raw.strip()))
            print(f"    OK: {len(raw):,} chars")
        except Exception as e:
            print(f"    WARN: no se pudo descargar {title}: {e}")

    return books


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build long-range dependency benchmark corpus from PG-19"
    )
    parser.add_argument("--out-dir", default=".",
                        help="Directorio de salida")
    parser.add_argument("--out-train", default="corpus_benchmark_train.txt")
    parser.add_argument("--out-val",   default="corpus_benchmark_val.txt")
    parser.add_argument("--ctx-len",   type=int, default=512)
    parser.add_argument("--batch-size",type=int, default=1)
    parser.add_argument("--n-train-books", type=int, default=50,
                        help="Libros para training (PG-19 train split)")
    parser.add_argument("--n-val-books",   type=int, default=10,
                        help="Libros para validación (PG-19 validation split)")
    parser.add_argument("--max-train-batches", type=int, default=2000,
                        help="Máximo de batches en train (limita tamaño)")
    parser.add_argument("--max-val-batches",   type=int, default=200,
                        help="Máximo de batches en val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-gutenberg", action="store_true",
                        help="Usar descarga directa de Gutenberg si HuggingFace no está disponible")
    args = parser.parse_args()

    out_train = os.path.join(args.out_dir, args.out_train)
    out_val   = os.path.join(args.out_dir, args.out_val)
    rng_train = random.Random(args.seed)
    rng_val   = random.Random(args.seed + 1)

    # --- Cargar libros ---
    use_hf = not args.fallback_gutenberg
    if use_hf:
        check_deps()

    print("\n=== AIDEEN Benchmark Corpus Builder ===")
    print(f"Fuente: {'PG-19 via HuggingFace' if use_hf else 'Project Gutenberg directo'}")
    print(f"ctx_len={args.ctx_len}, batch_size={args.batch_size}")
    chars_per_batch = args.ctx_len * args.batch_size * 4
    print(f"chars_per_batch≈{chars_per_batch:,} (~{args.ctx_len * args.batch_size:,} tokens)")

    print(f"\n[1/4] Cargando libros de TRAINING ({args.n_train_books} libros)...")
    if use_hf:
        train_books = load_pg19("train", args.n_train_books, args.seed)
    else:
        train_books = load_from_gutenberg(args.n_train_books, args.seed)

    print(f"\n[2/4] Cargando libros de VALIDACIÓN ({args.n_val_books} libros)...")
    if use_hf:
        # PG-19 validation split tiene solo 50 libros
        val_books = load_pg19("validation", min(args.n_val_books, 50), args.seed + 1)
    else:
        # Para fallback: usar los últimos libros de la lista
        val_books = load_from_gutenberg(
            min(args.n_val_books, len(GUTENBERG_URLS) - args.n_train_books),
            args.seed + 1
        )
        # Asegurarse de no solapar con train
        train_titles = {b[0] for b in train_books}
        val_books = [(t, x) for t, x in val_books if t not in train_titles]

    print(f"\n[3/4] Construyendo corpus de TRAINING...")
    train_text = build_corpus_from_books(
        train_books, args.ctx_len, args.batch_size, rng_train,
        max_batches=args.max_train_batches
    )
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        f.write(train_text)
    print(f"  Guardado: {out_train} ({os.path.getsize(out_train):,} bytes)")

    print(f"\n[4/4] Construyendo corpus de VALIDACIÓN...")
    val_text = build_corpus_from_books(
        val_books, args.ctx_len, args.batch_size, rng_val,
        max_batches=args.max_val_batches
    )
    with open(out_val, "w", encoding="utf-8") as f:
        f.write(val_text)
    print(f"  Guardado: {out_val} ({os.path.getsize(out_val):,} bytes)")

    # --- Resumen ---
    train_approx_tokens = os.path.getsize(out_train) // 4
    val_approx_tokens   = os.path.getsize(out_val) // 4

    print("\n=== Resumen ===")
    print(f"Training : {os.path.getsize(out_train):,} bytes ≈ {train_approx_tokens:,} tokens")
    print(f"Validación: {os.path.getsize(out_val):,} bytes ≈ {val_approx_tokens:,} tokens")
    print(f"Estructura: texto continuo por libro, {args.ctx_len}×{args.batch_size} tokens/batch")
    print(f"Separador de documento explícito: {DOC_MARKER}")
    print()
    print("Comandos para el benchmark:")
    print()
    print("# AIDEEN baseline:")
    print(f"AIDEEN_BATCH_SIZE={args.batch_size} AIDEEN_TRAIN_SEED=42 \\")
    print(f"cargo run --release --features wgpu -p aideen-training --bin train -- \\")
    print(f"  --file {out_train} --epochs 3")
    print()
    print("# Evaluar perplexity en val set:")
    print(f"cargo run --release --features wgpu -p aideen-training --bin eval -- \\")
    print(f"  --model model_large --file {out_val}")
    print()
    print("Referencia: PG-19 es el benchmark estándar para long-range LM.")
    print("Papers de referencia: Transformer-XL, Compressive Transformer, Mamba.")
    print("Perplexity en PG-19 val es la métrica principal comparable con literatura.")


if __name__ == "__main__":
    main()
