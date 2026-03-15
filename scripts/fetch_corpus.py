#!/usr/bin/env python3
"""
fetch_corpus.py -- Download open-license text and assemble a raw corpus.

Sources (in priority order):
  1. Existing corpus   : train_aideen.txt (~2 MB, SmolTalk + Rust docs)
  2. Project Gutenberg : ~20 public-domain books (capped at 200 KB each)
  3. Simple Wikipedia  : ~1000 random article extracts

Output: data/corpus/raw_corpus.txt  (target 50-100 MB)
"""

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "corpus"
OUTPUT_FILE  = OUTPUT_DIR / "raw_corpus.txt"

EXISTING_CORPUS = PROJECT_ROOT / "train_aideen.txt"

NET_TIMEOUT = 30
NET_RETRY   = 3
BOOK_CAP    = 200 * 1024  # 200 KB per book

DOC_SEP = "\n\n<|endoftext|>\n\n"

# Project Gutenberg book IDs and titles
GUTENBERG_BOOKS = [
    (1342,  "Pride and Prejudice"),
    (11,    "Alice's Adventures in Wonderland"),
    (1661,  "The Adventures of Sherlock Holmes"),
    (84,    "Frankenstein"),
    (1080,  "A Modest Proposal"),
    (98,    "A Tale of Two Cities"),
    (2701,  "Moby Dick"),
    (74,    "The Adventures of Tom Sawyer"),
    (1232,  "The Prince"),
    (46,    "A Christmas Carol"),
    (219,   "Heart of Darkness"),
    (345,   "Dracula"),
    (1400,  "Great Expectations"),
    (16328, "Beowulf"),
    (514,   "Little Women"),
    (2591,  "Grimms' Fairy Tales"),
    (5200,  "Metamorphosis"),
    (1952,  "The Yellow Wallpaper"),
    (36,    "The War of the Worlds"),
    (174,   "The Picture of Dorian Gray"),
]

# Simple Wikipedia config
WIKI_API = "https://simple.wikipedia.org/w/api.php"
WIKI_BATCH_SIZE = 20
WIKI_TARGET_ARTICLES = 1000

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str, retries: int = NET_RETRY,
              timeout: int = NET_TIMEOUT) -> bytes | None:
    """Download a URL with retries and exponential back-off."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AIDEEN-CorpusFetcher/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as exc:
            if attempt < retries - 1:
                time.sleep(2.0 ** attempt)
            else:
                print(f"  [WARN] Failed to fetch {url[:90]}: {exc}")
    return None


def fetch_text(url: str) -> str:
    data = fetch_url(url)
    return data.decode("utf-8", errors="replace") if data else ""


def fetch_json(url: str) -> dict | None:
    data = fetch_url(url)
    if data is None:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None

# ---------------------------------------------------------------------------
# 1. Existing corpus
# ---------------------------------------------------------------------------

def load_existing_corpus() -> str:
    """Read train_aideen.txt if it exists."""
    if not EXISTING_CORPUS.exists():
        print("  [INFO] No existing corpus found at train_aideen.txt, skipping.")
        return ""
    size = EXISTING_CORPUS.stat().st_size
    print(f"  Reading {EXISTING_CORPUS.name} ({size / 1_048_576:.1f} MB) ...")
    text = EXISTING_CORPUS.read_text(encoding="utf-8", errors="replace")
    print(f"  OK -- {len(text):,} chars")
    return text

# ---------------------------------------------------------------------------
# 2. Project Gutenberg
# ---------------------------------------------------------------------------

def strip_gutenberg_header_footer(text: str) -> str:
    """Remove the Project Gutenberg boilerplate header and footer."""
    # Find the start marker
    start_markers = ["*** START OF THIS PROJECT GUTENBERG",
                     "*** START OF THE PROJECT GUTENBERG",
                     "***START OF THIS PROJECT GUTENBERG",
                     "***START OF THE PROJECT GUTENBERG"]
    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Move past the marker line
            nl = text.find("\n", pos)
            start_idx = nl + 1 if nl != -1 else pos + len(marker)
            break

    # Find the end marker
    end_markers = ["*** END OF THIS PROJECT GUTENBERG",
                   "*** END OF THE PROJECT GUTENBERG",
                   "***END OF THIS PROJECT GUTENBERG",
                   "***END OF THE PROJECT GUTENBERG"]
    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break

    return text[start_idx:end_idx].strip()


def fetch_gutenberg_books() -> list[str]:
    """Download public-domain books from Project Gutenberg."""
    documents: list[str] = []
    total = len(GUTENBERG_BOOKS)

    for i, (book_id, title) in enumerate(GUTENBERG_BOOKS, 1):
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  [{i}/{total}] Gutenberg #{book_id}: {title} ...")

        raw = fetch_text(url)
        if not raw or len(raw) < 1000:
            print(f"    [WARN] Skipped (too short or download failed)")
            continue

        body = strip_gutenberg_header_footer(raw)
        if len(body) < 500:
            print(f"    [WARN] Skipped (body too short after stripping)")
            continue

        # Cap at BOOK_CAP bytes (measured in UTF-8)
        if len(body.encode("utf-8")) > BOOK_CAP:
            # Truncate at a word boundary near the cap
            body_bytes = body.encode("utf-8")[:BOOK_CAP]
            body = body_bytes.decode("utf-8", errors="ignore").rsplit(" ", 1)[0]

        documents.append(f"# {title}\n# Source: Project Gutenberg #{book_id} (public domain)\n\n{body}")
        print(f"    OK -- {len(body):,} chars")
        time.sleep(0.3)  # be polite

    print(f"  Gutenberg: {len(documents)}/{total} books downloaded")
    return documents

# ---------------------------------------------------------------------------
# 3. Simple English Wikipedia
# ---------------------------------------------------------------------------

def fetch_wiki_articles() -> list[str]:
    """Fetch random articles from Simple English Wikipedia."""
    documents: list[str] = []
    fetched_titles: set[str] = set()
    batches_needed = (WIKI_TARGET_ARTICLES + WIKI_BATCH_SIZE - 1) // WIKI_BATCH_SIZE

    print(f"  Fetching ~{WIKI_TARGET_ARTICLES} articles in batches of {WIKI_BATCH_SIZE} ...")

    for batch_num in range(batches_needed):
        if len(documents) >= WIKI_TARGET_ARTICLES:
            break

        # Step 1: get random page titles
        random_url = (
            f"{WIKI_API}?action=query&list=random"
            f"&rnlimit={WIKI_BATCH_SIZE}&rnnamespace=0&format=json"
        )
        random_data = fetch_json(random_url)
        if random_data is None:
            print(f"    [WARN] Batch {batch_num + 1}: failed to get random titles")
            time.sleep(1.0)
            continue

        pages = random_data.get("query", {}).get("random", [])
        if not pages:
            continue

        # Deduplicate
        titles = [p["title"] for p in pages if p["title"] not in fetched_titles]
        if not titles:
            continue
        fetched_titles.update(titles)

        # Step 2: get extracts for these titles
        titles_param = "|".join(titles)
        extract_url = (
            f"{WIKI_API}?action=query&titles={urllib.request.quote(titles_param)}"
            f"&prop=extracts&explaintext=1&exlimit={WIKI_BATCH_SIZE}&format=json"
        )
        extract_data = fetch_json(extract_url)
        if extract_data is None:
            print(f"    [WARN] Batch {batch_num + 1}: failed to get extracts")
            time.sleep(1.0)
            continue

        query_pages = extract_data.get("query", {}).get("pages", {})
        batch_count = 0
        for page_id, page_info in query_pages.items():
            if page_id == "-1":
                continue
            title = page_info.get("title", "")
            extract = page_info.get("extract", "")
            if not extract or len(extract) < 200:
                continue  # skip stubs

            doc = f"# {title}\n# Source: Simple English Wikipedia (CC BY-SA)\n\n{extract}"
            documents.append(doc)
            batch_count += 1

        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            print(f"    Batch {batch_num + 1}/{batches_needed}: "
                  f"{len(documents)} articles so far")

        time.sleep(0.5)  # rate-limit

    print(f"  Wikipedia: {len(documents)} articles fetched")
    return documents

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  AIDEEN Corpus Fetcher")
    print("=" * 64)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_parts: list[str] = []

    # 1. Existing corpus
    print("[1/3] Existing corpus (train_aideen.txt)")
    existing = load_existing_corpus()
    if existing:
        all_parts.append(existing)
    print()

    # 2. Project Gutenberg
    print("[2/3] Project Gutenberg (public domain books)")
    gutenberg_docs = fetch_gutenberg_books()
    for doc in gutenberg_docs:
        all_parts.append(doc)
    print()

    # 3. Simple English Wikipedia
    print("[3/3] Simple English Wikipedia")
    wiki_docs = fetch_wiki_articles()
    for doc in wiki_docs:
        all_parts.append(doc)
    print()

    # Assemble
    print("Assembling corpus ...")
    corpus = DOC_SEP.join(all_parts)

    OUTPUT_FILE.write_text(corpus, encoding="utf-8")
    size_bytes = OUTPUT_FILE.stat().st_size
    size_mb = size_bytes / 1_048_576

    print(f"Wrote {OUTPUT_FILE}")
    print(f"  Total size : {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  Documents  : {len(all_parts):,}")
    print()
    print("Next step: python scripts/tokenize_corpus.py")


if __name__ == "__main__":
    main()
