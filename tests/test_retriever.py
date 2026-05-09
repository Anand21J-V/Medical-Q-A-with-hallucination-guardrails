"""
tests/test_retriever.py

Unit tests for BookChunker — no ChromaDB required.
"""

import pytest
from src.ingestion.chunker import BookChunker
from src.ingestion.document_loader import RawPage


def make_page(text: str, page: int = 1) -> RawPage:
    return RawPage(
        text=text,
        source="test_book.pdf",
        page_number=page,
        metadata={"format": "pdf"},
    )


class TestBookChunker:
    def test_short_text_single_chunk(self):
        chunker = BookChunker()
        page = make_page("This is a short medical sentence about hypertension.")
        chunks = chunker.chunk_pages([page])
        assert len(chunks) == 1
        assert "hypertension" in chunks[0].text

    def test_long_text_multiple_chunks(self):
        chunker = BookChunker()
        # Generate a text longer than chunk_size (800 chars)
        long_text = ("Diabetes mellitus is a chronic condition. " * 50)
        page = make_page(long_text)
        chunks = chunker.chunk_pages([page])
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        chunker = BookChunker()
        page = make_page("Sample medical text about cardiology.", page=7)
        chunks = chunker.chunk_pages([page])
        assert chunks[0].source == "test_book.pdf"
        assert chunks[0].page_number == 7

    def test_empty_pages_skipped(self):
        chunker = BookChunker()
        empty_page = make_page("   \n\n   ")
        chunks = chunker.chunk_pages([empty_page])
        assert len(chunks) == 0

    def test_multiple_pages(self):
        chunker = BookChunker()
        pages = [make_page(f"Page {i} content about medicine." * 10, page=i) for i in range(1, 6)]
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= 5
        sources = {c.source for c in chunks}
        assert sources == {"test_book.pdf"}
