#!/usr/bin/env python3
"""
fetch_corpus.py -- Download open-license text corpus for AIDEEN training.

Sources:
  1. Existing corpus   : train_aideen.txt (~2 MB, SmolTalk + Rust docs)
  2. Wikipedia dumps    : Simple English + Spanish Wikipedia (articles XML)
  3. Project Gutenberg  : ~100 public-domain books

Output: data/corpus/raw_corpus.txt  (target ~2-5 GB)

All data is public domain or CC BY-SA (legal for training).
"""

import bz2
import json
import os
import re
import struct
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path("data/corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_FILE = OUT_DIR / "raw_corpus.txt"


def download_file(url, dest, desc=""):
    """Download a file with progress reporting."""
    if dest.exists():
        size = dest.stat().st_size
        print(f"  [skip] {desc or dest.name} already exists ({size/1e6:.1f} MB)")
        return True
    print(f"  Downloading {desc or url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AideenCorpusFetcher/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded/1e6:.1f} MB", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_wiki_articles(bz2_path, max_articles=None):
    """Extract article text from a Wikipedia XML dump (bz2 compressed)."""
    print(f"  Extracting articles from {bz2_path.name}...")
    articles = []
    count = 0
    in_text = False
    current_text = []
    current_title = ""

    with bz2.open(bz2_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Simple XML parsing (avoids heavy deps)
            if "<title>" in line:
                m = re.search(r"<title>(.*?)</title>", line)
                if m:
                    current_title = m.group(1)

            if "<text" in line:
                in_text = True
                # Get text after the tag on same line
                m = re.search(r'<text[^>]*>(.*)', line)
                if m:
                    current_text = [m.group(1)]
                else:
                    current_text = []
                continue

            if in_text:
                if "</text>" in line:
                    current_text.append(line.split("</text>")[0])
                    in_text = False
                    text = "\n".join(current_text)

                    # Skip redirects, stubs, meta pages
                    if text.startswith("#REDIRECT") or text.startswith("#redirect"):
                        continue
                    if len(text) < 500:
                        continue
                    if ":" in current_title:  # Skip Wikipedia:, Template:, etc.
                        continue

                    # Strip basic wiki markup
                    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)  # [[link|text]] -> text
                    text = re.sub(r"'{2,5}", "", text)  # Bold/italic markup
                    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Templates
                    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)  # References
                    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
                    text = re.sub(r'\n{3,}', '\n\n', text)  # Excess newlines

                    if len(text) > 200:
                        articles.append(f"# {current_title}\n\n{text}")
                        count += 1
                        if count % 10000 == 0:
                            print(f"    {count} articles extracted...")
                        if max_articles and count >= max_articles:
                            break
                else:
                    current_text.append(line)

    print(f"    Total: {count} articles")
    return articles


def fetch_wikipedia():
    """Download and extract Wikipedia articles."""
    wiki_sources = [
        {
            "name": "Simple English Wikipedia",
            "url": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
            "file": "simplewiki-latest.xml.bz2",
            "max": None,  # All articles (~200K)
        },
        # English Wikipedia is ~22GB compressed. Download separately with:
        #   wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -O data/corpus/enwiki-latest.xml.bz2
        # Then re-run this script to extract it.
        # {
        #     "name": "English Wikipedia",
        #     "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        #     "file": "enwiki-latest.xml.bz2",
        #     "max": None,
        # },
        {
            "name": "Spanish Wikipedia",
            "url": "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2",
            "file": "eswiki-latest.xml.bz2",
            "max": None,  # All articles
        },
    ]

    total_bytes = 0
    for src in wiki_sources:
        print(f"\n{'='*60}")
        print(f"Source: {src['name']}")
        print(f"{'='*60}")
        dest = OUT_DIR / src["file"]
        if download_file(src["url"], dest, src["name"]):
            articles = extract_wiki_articles(dest, src.get("max"))
            # Write incrementally to avoid holding everything in memory
            with open(CHUNK_FILE, "a", encoding="utf-8") as f:
                for article in articles:
                    f.write(article)
                    f.write("\n\n")
                    total_bytes += len(article) + 2
            print(f"  Written: {total_bytes/1e6:.1f} MB cumulative")
    return total_bytes


def fetch_gutenberg():
    """Download public-domain books from Project Gutenberg."""
    print(f"\n{'='*60}")
    print(f"Source: Project Gutenberg")
    print(f"{'='*60}")

    # Top 100 most downloaded + classics
    book_ids = [
        1342, 11, 1661, 84, 98, 2701, 1232, 174, 345, 5200,
        1400, 16, 43, 76, 1952, 219, 2591, 2554, 1080, 74,
        996, 55, 160, 1260, 844, 768, 2600, 4300, 25344, 1184,
        2542, 46, 514, 1023, 2814, 35, 6130, 135, 158, 203,
        3207, 1497, 100, 244, 2852, 120, 730, 205, 36, 209,
        33, 1322, 113, 23, 408, 30254, 1727, 2148, 105, 2500,
        4363, 27827, 161, 1399, 3296, 3825, 829, 910, 3090, 19942,
        28054, 932, 2680, 1998, 2097, 14838, 19033, 41, 45, 947,
        58585, 28885, 1250, 5740, 2641, 7370, 236, 766, 2160, 541,
        1257, 521, 375, 158, 30, 394, 420, 8800, 2197, 6761,
    ]

    texts = []
    for bid in book_ids:
        urls = [
            f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AideenCorpusFetcher/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                # Strip Gutenberg header/footer
                start = text.find("*** START")
                if start < 0:
                    start = text.find("***START")
                end = text.find("*** END")
                if end < 0:
                    end = text.find("***END")
                if start > 0 and end > start:
                    text = text[start + 80:end]
                if len(text) > 1000:
                    texts.append(text)
                    print(f"  Book {bid}: {len(text)/1e3:.0f} KB")
                    break
            except Exception:
                continue
        time.sleep(0.3)  # Be polite to Gutenberg servers

    result = "\n\n".join(texts)
    print(f"  Total Gutenberg: {len(result)/1e6:.1f} MB ({len(texts)} books)")
    return result


def main():
    print("AIDEEN Corpus Fetcher")
    print(f"Output: {CHUNK_FILE.absolute()}")
    print(f"Storage: {OUT_DIR.absolute()}")
    print()

    # Start fresh
    if CHUNK_FILE.exists():
        CHUNK_FILE.unlink()

    # 1. Existing corpus
    existing = Path("train_aideen.txt")
    if existing.exists():
        text = existing.read_text(errors="replace")
        with open(CHUNK_FILE, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n\n")
        print(f"Existing corpus: {len(text)/1e6:.1f} MB")

    # 2. Wikipedia (writes incrementally to CHUNK_FILE)
    wiki_bytes = fetch_wikipedia()

    # 3. Gutenberg (append)
    gut_text = fetch_gutenberg()
    if gut_text:
        with open(CHUNK_FILE, "a", encoding="utf-8") as f:
            f.write(gut_text)

    # Report
    total_size = CHUNK_FILE.stat().st_size if CHUNK_FILE.exists() else 0
    print(f"\n{'='*60}")
    print(f"CORPUS COMPLETE")
    print(f"{'='*60}")
    print(f"  Total size: {total_size/1e6:.1f} MB ({total_size/1e9:.2f} GB)")
    print(f"  Output: {CHUNK_FILE.absolute()}")
    print(f"  Approx tokens (at 4 chars/token): ~{total_size//4:,}")


if __name__ == "__main__":
    main()
