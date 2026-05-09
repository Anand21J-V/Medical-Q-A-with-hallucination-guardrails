"""
src/ingestion/chunker.py

Splits raw document pages into fixed-size overlapping chunks suitable
for dense retrieval. Uses LangChain's RecursiveCharacterTextSplitter with
a separator hierarchy that respects paragraph and sentence boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.document_loader import RawPage
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextChunk:
    """A single chunk ready for embedding and storage."""

    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: dict


class BookChunker:
    """
    Chunks RawPage objects into overlapping text segments.

    Strategy: RecursiveCharacterTextSplitter with a separator hierarchy
    that prefers paragraph breaks → sentences → words, minimising mid-sentence cuts.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(self) -> None:
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
        )

    def chunk_pages(self, pages: Sequence[RawPage]) -> list[TextChunk]:
        """Convert a list of RawPage objects into a flat list of TextChunks."""
        all_chunks: list[TextChunk] = []

        for page in pages:
            raw_chunks = self._splitter.split_text(page.text)
            for idx, chunk_text in enumerate(raw_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text or len(chunk_text) < 30:
                    continue
                all_chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=idx,
                        metadata={
                            **page.metadata,
                            "page_number": page.page_number,
                            "chunk_index": idx,
                            "source": page.source,
                        },
                    )
                )

        logger.info(
            "chunking_complete",
            input_pages=len(pages),
            output_chunks=len(all_chunks),
            avg_chars=round(
                sum(len(c.text) for c in all_chunks) / max(len(all_chunks), 1)
            ),
        )
        return all_chunks
