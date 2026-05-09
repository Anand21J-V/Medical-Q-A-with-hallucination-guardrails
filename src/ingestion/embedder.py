"""
src/ingestion/embedder.py

Embeds TextChunks using a local sentence-transformer model and upserts
them into a persistent ChromaDB collection.

Design decisions:
- Local embeddings (no API cost, no latency on embed calls)
- Deterministic chunk IDs prevent duplicate inserts on re-ingestion
- Batch upsert for efficiency
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.ingestion.chunker import TextChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 100


def _chunk_id(chunk: TextChunk) -> str:
    """Deterministic ID: SHA-256 of (source + page + chunk_index)."""
    key = f"{chunk.source}::{chunk.page_number}::{chunk.chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Embedder:
    """
    Manages the ChromaDB collection lifecycle and chunk upsert.
    Thread-safe for single-process use (ChromaDB's PersistentClient is not
    multiprocess-safe — use a ChromaDB server in production).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "embedder_initialized",
            collection=settings.collection_name,
            existing_docs=self._collection.count(),
        )

    def upsert(self, chunks: Sequence[TextChunk]) -> int:
        """
        Embed and upsert chunks in batches.
        Returns the number of chunks successfully upserted.
        """
        if not chunks:
            logger.warning("upsert_called_with_empty_chunks")
            return 0

        total = 0
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            ids = [_chunk_id(c) for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info(
                "batch_upserted",
                batch=i // _BATCH_SIZE + 1,
                size=len(batch),
                total_so_far=total,
            )

        logger.info(
            "upsert_complete",
            chunks_upserted=total,
            collection_size=self._collection.count(),
        )
        return total

    def collection_size(self) -> int:
        return self._collection.count()
