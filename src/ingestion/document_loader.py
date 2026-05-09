"""
src/ingestion/document_loader.py

Loads raw text from medical books in PDF, EPUB, or plain-text format.
Returns a list of RawPage objects preserving source metadata.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawPage:
    """A single logical page/section of text with source metadata."""

    text: str
    source: str
    page_number: int
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """
    Unified loader for PDF, EPUB, and plain-text documents.
    Raises ValueError for unsupported formats.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md"}

    def load(self, path: str | Path) -> list[RawPage]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(
            "loading_document",
            path=str(file_path),
            format=ext,
            size_mb=round(file_path.stat().st_size / 1_048_576, 2),
        )

        if ext == ".pdf":
            pages = list(self._load_pdf(file_path))
        elif ext == ".epub":
            pages = list(self._load_epub(file_path))
        else:
            pages = list(self._load_text(file_path))

        logger.info(
            "document_loaded",
            path=str(file_path),
            pages=len(pages),
            total_chars=sum(len(p.text) for p in pages),
        )
        return pages

    # ── Private loaders ───────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> Iterator[RawPage]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("Install pymupdf: pip install pymupdf") from exc

        doc = fitz.open(str(path))
        source = path.name

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            yield RawPage(
                text=text,
                source=source,
                page_number=page_num,
                metadata={"file": str(path), "format": "pdf"},
            )
        doc.close()

    def _load_epub(self, path: Path) -> Iterator[RawPage]:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "Install ebooklib and beautifulsoup4: "
                "pip install ebooklib beautifulsoup4"
            ) from exc

        book = epub.read_epub(str(path))
        source = path.name
        page_num = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            if not text or len(text) < 50:
                continue
            page_num += 1
            yield RawPage(
                text=text,
                source=source,
                page_number=page_num,
                metadata={
                    "file": str(path),
                    "format": "epub",
                    "item_id": item.id,
                },
            )

    def _load_text(self, path: Path) -> Iterator[RawPage]:
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.name

        # Split plain text into ~500-line logical pages
        lines = text.splitlines()
        page_size = 500
        for page_num, start in enumerate(
            range(0, len(lines), page_size), start=1
        ):
            chunk = "\n".join(lines[start : start + page_size]).strip()
            if not chunk:
                continue
            yield RawPage(
                text=chunk,
                source=source,
                page_number=page_num,
                metadata={"file": str(path), "format": "text"},
            )
