#!/usr/bin/env python3
"""
scripts/ingest_book.py

CLI for ingesting a medical book into ChromaDB.

Usage:
    python scripts/ingest_book.py data/raw/harrison_principles.pdf
    python scripts/ingest_book.py data/raw/medbook.epub --collection my_collection
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import BookChunker
from src.ingestion.embedder import Embedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a medical book into the CRAG vector store."
    )
    parser.add_argument("path", type=str, help="Path to the book file (PDF/EPUB/TXT)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk the file without writing to ChromaDB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.path)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Medical CRAG — Book Ingestion")
    print(f"{'='*60}")
    print(f"  File  : {file_path.name}")
    print(f"  Format: {file_path.suffix.upper()}")
    print(f"  Size  : {file_path.stat().st_size / 1_048_576:.2f} MB")
    print(f"{'='*60}\n")

    # Step 1: Load
    print("Step 1/3 — Loading document...")
    loader = DocumentLoader()
    pages = loader.load(file_path)
    print(f"  → {len(pages)} pages loaded")

    # Step 2: Chunk
    print("Step 2/3 — Chunking...")
    chunker = BookChunker()
    chunks = chunker.chunk_pages(pages)
    total_chars = sum(len(c.text) for c in chunks)
    print(f"  → {len(chunks)} chunks created")
    print(f"  → Avg chunk size: {total_chars // max(len(chunks), 1)} chars")

    if args.dry_run:
        print("\n[DRY RUN] Skipping ChromaDB upsert.")
        print(f"\nSample chunk (chunk 0):\n{'-'*40}")
        print(chunks[0].text[:300] if chunks else "(no chunks)")
        return

    # Step 3: Embed + Upsert
    print("Step 3/3 — Embedding and upserting to ChromaDB...")
    print("  (This may take a few minutes for large books...)")
    embedder = Embedder()
    upserted = embedder.upsert(chunks)

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"  Ingestion complete in {elapsed}s")
    print(f"  Chunks upserted   : {upserted}")
    print(f"  Collection size   : {embedder.collection_size()} total chunks")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
