#!/usr/bin/env python3
"""
build_dataset.py — Construye el dataset de entrenamiento AIDEEN Genesis v3.

Fuentes:
  1. The Rust Book completo (GitHub CC BY 4.0)  — lenguaje base
  2. Papers arXiv (ar5iv HTML)                  — arquitectura: DEQ, Mamba, SSM,
                                                   Transformers, AdamW, federated...

Output: aideen-backbone/dataset_genesis.txt
"""

import re
import json
import time
import random
import urllib.request
import urllib.error
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUTPUT_FILE  = PROJECT_ROOT / "aideen-backbone" / "dataset_genesis.txt"
SEED         = 42
DOC_SEP      = "<|endoftext|>\n"   # Separador GPT-2 #50256
NET_TIMEOUT  = 40
NET_RETRY    = 3
AR5IV_DELAY  = 1.5   # segundos entre peticiones a ar5iv

random.seed(SEED)

# ── Papers arXiv ──────────────────────────────────────────────────────────────
# Todos verificados en ar5iv (>10K chars de texto útil)
ARXIV_PAPERS = {
    # ── Arquitectura propia de AIDEEN: DEQ + SSM ──
    "1909.01377": "Deep Equilibrium Models (DEQ) — Bai et al. 2019",
    "2102.07074": "Efficient Training of Deep Equilibrium Models — Bai et al. 2021",
    "2111.00396": "Efficiently Modeling Long Sequences with Structured State Spaces (S4) — Gu et al. 2022",
    "2005.08649": "HiPPO: Recurrent Memory with Optimal Polynomial Projections — Gu et al. 2020",
    "2208.04933": "Diagonal State Spaces are as Effective as Structured State Spaces — Gupta et al. 2022",
    "2212.14052": "Hungry Hungry Hippos (H3): Language Modeling with State Space Models — Fu et al. 2023",
    "2206.11893": "On the Parameterization and Initialization of Diagonal State Space Models — Gupta et al. 2022",
    "2110.13985": "Combining Recurrent, Convolutional, and Continuous-time Models (LRU/LSSL) — Gu et al. 2022",
    # ── Transformers y atención ──
    "1706.03762": "Attention Is All You Need (Transformer) — Vaswani et al. 2017",
    "2205.14135": "FlashAttention: Fast and Memory-Efficient Exact Attention — Dao et al. 2022",
    "1810.04805": "BERT: Pre-training Deep Bidirectional Transformers — Devlin et al. 2019",
    # ── Optimización y entrenamiento ──
    "1711.05101": "Decoupled Weight Decay Regularization (AdamW) — Loshchilov & Hutter 2019",
    "1607.06450": "Layer Normalization — Ba et al. 2016",
    "1502.03167": "Batch Normalization — Ioffe & Szegedy 2015",
    "1512.03385": "Deep Residual Learning for Image Recognition (ResNet) — He et al. 2016",
    # ── Tokenización y lenguaje ──
    "1808.06226": "SentencePiece: Subword Tokenization — Kudo & Richardson 2018",
    "1301.3666":  "Efficient Estimation of Word Representations (Word2Vec) — Mikolov et al. 2013",
    # ── AI Distribuida y Federated Learning ──
    "1602.05629": "Communication-Efficient Learning of Deep Networks (Federated Learning) — McMahan et al. 2017",
    "1910.06054": "Advances and Open Problems in Federated Learning — Kairouz et al. 2021",
}

# ── Capítulos del Rust Book ───────────────────────────────────────────────────
RUST_BOOK_BASE    = "https://raw.githubusercontent.com/rust-lang/book/main/src/"
RUST_BOOK_SUMMARY = RUST_BOOK_BASE + "SUMMARY.md"

# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades de red
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_url(url: str, retries: int = NET_RETRY, timeout: int = NET_TIMEOUT) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIDEEN-DatasetBuilder/3.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2.0 ** attempt)
            else:
                print(f"    ⚠️  Error ({e.__class__.__name__}): {url[:80]}")
    return None


def fetch_text(url: str) -> str:
    data = fetch_url(url)
    return data.decode("utf-8", errors="replace") if data else ""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. THE RUST BOOK (GitHub, CC BY 4.0)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_rust_book() -> list[dict]:
    print("  📥 Descargando índice del Rust Book...")
    summary = fetch_text(RUST_BOOK_SUMMARY)
    if not summary:
        print("  ⚠️  No se pudo obtener SUMMARY.md")
        return []

    chapter_files = re.findall(r'\(([^)]+\.md)\)', summary)
    chapter_files = [f for f in chapter_files if not f.startswith("http")]
    chapter_files = list(dict.fromkeys(chapter_files))

    documents = []
    print(f"  📥 Descargando {len(chapter_files)} capítulos...")

    for filename in chapter_files:
        text = fetch_text(RUST_BOOK_BASE + filename)
        if not text or len(text) < 100:
            continue

        text = re.sub(r'\{\{#rustdoc_include [^}]+\}\}', '[ver código en rustdoc]', text)
        text = re.sub(r'\{\{#include [^}]+\}\}',         '[ver ejemplo]',           text)
        text = re.sub(r'<!-- .+? -->',                   '',   text, flags=re.DOTALL)
        text = text.strip()

        title_match = re.search(r'^#+ (.+)', text, re.MULTILINE)
        title = title_match.group(1) if title_match else filename

        body = f"# The Rust Programming Language — {title}\n\n{text}"
        documents.append({"source": f"RustBook:{filename}", "text": body})
        time.sleep(0.1)

    print(f"  ✓ Rust Book: {len(documents)} capítulos")
    return documents

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PAPERS ARXIV (via ar5iv HTML → texto limpio)
# ═══════════════════════════════════════════════════════════════════════════════

def html_to_text(html: str) -> str:
    """Convierte HTML de ar5iv a texto plano limpio."""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>',   '', html, flags=re.DOTALL)
    html = re.sub(r'<nav[^>]*>.*?</nav>',        '', html, flags=re.DOTALL)
    html = re.sub(r'<header[^>]*>.*?</header>',  '', html, flags=re.DOTALL)
    html = re.sub(r'<footer[^>]*>.*?</footer>',  '', html, flags=re.DOTALL)
    html = re.sub(r'<br\s*/?>', '\n', html)
    html = re.sub(r'<p[^>]*>',  '\n', html)
    html = re.sub(r'</p>',      '\n', html)
    html = re.sub(r'<h([1-6])[^>]*>', lambda m: '\n' + '#' * int(m.group(1)) + ' ', html)
    html = re.sub(r'</h[1-6]>', '\n', html)
    html = re.sub(r'<li[^>]*>', '\n- ', html)
    html = re.sub(r'<tr[^>]*>', '\n',  html)
    html = re.sub(r'<td[^>]*>', ' | ', html)
    html = re.sub(r'<th[^>]*>', ' | ', html)
    html = re.sub(r'<[^>]+>', '', html)
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    html = html.replace('&mdash;', '—').replace('&ndash;', '–').replace('&hellip;', '...')
    html = re.sub(r'[ \t]{2,}', ' ', html)
    html = re.sub(r'\n{4,}',    '\n\n\n', html)
    return html.strip()


def fetch_arxiv_paper(arxiv_id: str, title: str) -> dict | None:
    """Descarga un paper de arXiv vía ar5iv (HTML) y lo convierte a texto."""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    data = fetch_url(url, timeout=60)
    if not data:
        return None

    html = data.decode("utf-8", errors="replace")
    if len(html) < 5000:
        print(f"    ⚠️  Contenido muy corto para {arxiv_id}")
        return None

    text = html_to_text(html)

    # Recortar navegación inicial — empezar desde Abstract o Introduction
    for marker in ["## Abstract", "# Abstract", "Abstract\n", "Introduction\n"]:
        pos = text.find(marker)
        if 0 < pos < 5000:
            text = text[pos:]
            break

    if len(text) < 2000:
        print(f"    ⚠️  Texto extraído muy corto para {arxiv_id}")
        return None

    body = (
        f"# {title}\n"
        f"# ArXiv: {arxiv_id}\n"
        f"# Fuente: arXiv (acceso abierto)\n\n"
        f"{text}"
    )
    return {"source": f"arXiv:{arxiv_id}", "text": body}


def fetch_all_papers(papers: dict) -> list[dict]:
    documents = []
    total = len(papers)
    for i, (arxiv_id, title) in enumerate(papers.items(), 1):
        print(f"  📥 [{i}/{total}] arXiv:{arxiv_id} — {title[:60]}...")
        doc = fetch_arxiv_paper(arxiv_id, title)
        if doc:
            documents.append(doc)
            print(f"    ✓ {len(doc['text']):,} chars")
        time.sleep(AR5IV_DELAY)

    print(f"  ✓ Papers arXiv: {len(documents)}/{total} descargados")
    return documents

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENSAMBLAR Y ESCRIBIR
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_write(documents: list[dict], output: Path) -> None:
    random.shuffle(documents)

    total_chars = sum(len(d["text"]) for d in documents)
    print(f"\n  📊 Total: {len(documents)} documentos, {total_chars:,} chars")
    print(f"  📊 Tokens estimados: ~{total_chars // 2:,}")

    source_counts: dict[str, int] = {}
    for d in documents:
        src = d["source"].split(":")[0]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        chars = sum(len(d["text"]) for d in documents if d["source"].startswith(src))
        pct   = 100 * chars / total_chars
        print(f"    {src:20s}: {cnt:4d} docs  {chars:>9,} chars  ({pct:5.1f}%)")

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✍️  Escribiendo {output} ...")
    with output.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc["text"])
            f.write("\n\n" + DOC_SEP + "\n")

    size_mb = output.stat().st_size / 1_048_576
    print(f"  ✓ Dataset escrito: {size_mb:.2f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AIDEEN Dataset Builder v3  —  Genesis con papers arXiv     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    all_docs: list[dict] = []

    # 1. Rust Book
    print("① The Rust Programming Language Book...")
    all_docs.extend(fetch_rust_book())

    # 2. Papers arXiv
    print(f"\n② Papers arXiv ({len(ARXIV_PAPERS)} papers)...")
    all_docs.extend(fetch_all_papers(ARXIV_PAPERS))

    # 3. Ensamblar
    print("\n③ Ensamblando dataset...")
    build_and_write(all_docs, OUTPUT_FILE)

    print()
    print("✅ Dataset listo. Siguiente paso:")
    print(f"   cargo run --release --features wgpu -p aideen-training --bin train -- \\")
    print(f"     --file aideen-backbone/dataset_genesis.txt")


if __name__ == "__main__":
    main()
