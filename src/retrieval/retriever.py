"""
src/retrieval/retriever.py

Performs cosine-similarity retrieval against the ChromaDB collection.
Returns a list of RetrievedChunk objects with text, metadata, and distance score.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document chunk with its distance score."""

    id: str
    text: str
    source: str
    page_number: int
    distance: float  # cosine distance: lower = more similar

    @property
    def similarity(self) -> float:
        """Convert cosine distance to similarity score (0–1)."""
        return round(1.0 - self.distance, 4)


class Retriever:
    """
    Wraps ChromaDB query with consistent return types.
    Lazily initialises the client on first call (avoids startup overhead).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._settings.embedding_model
            )
            client = chromadb.PersistentClient(path=self._settings.chroma_path)
            self._collection = client.get_or_create_collection(
                name=self._settings.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query: Natural-language question.
            top_k: Override the default TOP_K setting.

        Returns:
            List of RetrievedChunk, sorted by similarity descending.
        """
        k = top_k or self._settings.top_k
        collection = self._get_collection()
        doc_count = collection.count()

        if doc_count == 0:
            logger.warning(
                "retrieve_on_empty_collection",
                collection=self._settings.collection_name,
            )
            return []

        k = min(k, doc_count)
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for idx in range(len(results["ids"][0])):
            meta = results["metadatas"][0][idx] or {}
            chunks.append(
                RetrievedChunk(
                    id=results["ids"][0][idx],
                    text=results["documents"][0][idx],
                    source=meta.get("source", "unknown"),
                    page_number=int(meta.get("page_number", 0)),
                    distance=results["distances"][0][idx],
                )
            )

        logger.info(
            "retrieval_complete",
            query_preview=query[:60],
            k=k,
            top_similarity=chunks[0].similarity if chunks else None,
        )
        return chunks
