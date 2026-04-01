#!/usr/bin/env python3
"""
Build reproducible corpora for AIDEEN training.

Outputs:
  - corpus_pretrain_minimal.txt: clean pretraining text
  - corpus_chat_instruct.txt: chat/instruction text kept separate

The current combined corpus is known to contain a large chat block between two
<|endoftext|> separators. For pretraining we keep the clean book/web segments
and append a small set of local project documents with architecture knowledge.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path("/Users/sergiosolis/Programacion/AIDEEN")
DEFAULT_SOURCE = Path("/Users/sergiosolis/Programacion/aideen/corpus_combined.txt")
DEFAULT_OUTPUT_DIR = Path("/Users/sergiosolis/Programacion/aideen")
SEP = "<|endoftext|>"

KNOWLEDGE_FILES = [
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "PLAN.md",
    ROOT / "ARCHITECTURE_DECISIONS.md",
    ROOT / "docs" / "distributed_training_users.md",
]


def normalize_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text + "\n"


def tagged_doc(path: Path) -> str:
    body = normalize_block(path.read_text(errors="ignore"))
    return f"# SOURCE: {path.name}\n\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare clean pretrain/chat corpora for AIDEEN")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pretrain-name",
        default="corpus_pretrain_minimal.txt",
        help="Filename for the clean pretraining corpus",
    )
    parser.add_argument(
        "--chat-name",
        default="corpus_chat_instruct.txt",
        help="Filename for the chat/instruction corpus",
    )
    args = parser.parse_args()

    source = args.source
    if not source.exists():
        print(f"ERROR: source corpus not found: {source}", file=sys.stderr)
        return 1

    text = source.read_text(errors="ignore")
    parts = text.split(SEP)
    if len(parts) < 3:
        print(
            f"ERROR: expected at least 3 segments separated by {SEP!r}, got {len(parts)}",
            file=sys.stderr,
        )
        return 2

    rust_book = normalize_block(parts[0])
    chat_block = normalize_block(parts[1])
    fineweb = normalize_block(SEP.join(parts[2:]))

    knowledge_docs = [tagged_doc(path) for path in KNOWLEDGE_FILES if path.exists()]

    pretrain_sections = [rust_book, fineweb, *knowledge_docs]
    pretrain_text = f"\n{SEP}\n".join(pretrain_sections).strip() + "\n"
    chat_text = chat_block

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = out_dir / args.pretrain_name
    chat_path = out_dir / args.chat_name

    pretrain_path.write_text(pretrain_text)
    chat_path.write_text(chat_text)

    for label, payload, path in [
        ("pretrain", pretrain_text, pretrain_path),
        ("chat", chat_text, chat_path),
    ]:
        print(
            f"{label}: {path} | chars={len(payload)} | bytes={len(payload.encode('utf-8'))} | "
            f"USER:={payload.count('USER:')} ASSISTANT:={payload.count('ASSISTANT:')} SYSTEM:={payload.count('SYSTEM:')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
