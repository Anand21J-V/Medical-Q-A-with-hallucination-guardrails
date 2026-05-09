"""
src/api/routes.py

FastAPI route handlers.
Each handler is thin: validate → delegate → return schema.
All heavy logic lives in graph/ or src/ modules.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from graph.crag_graph import crag_graph
from graph.state import CRAGState
from src.api.schemas import (
    AskRequest,
    AskResponse,
    AuditResponse,
    ChunkTrace,
    HealthResponse,
    IngestResponse,
)
from src.ingestion.chunker import BookChunker
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.embedder import Embedder
from src.utils.audit_logger import get_audit_logger
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md"}


@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask(body: AskRequest) -> AskResponse:
    """
    Submit a medical question. The CRAG graph handles retrieval,
    grading, optional web search, and answer generation.
    """
    request_id = str(uuid.uuid4())
    t0 = time.time()

    logger.info("ask_request", request_id=request_id, question=body.question[:80])

    initial_state: CRAGState = {
        "question": body.question,
        "request_id": request_id,
        "_start_time": t0,
    }

    try:
        final_state: CRAGState = crag_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("crag_graph_error", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    result = final_state.get("result")
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph completed without a result.",
        )

    crag_trace = [
        ChunkTrace(
            text_preview=c.text[:120],
            source=c.source,
            page_number=c.page_number,
            relevance_score=c.relevance_score,
        )
        for c in final_state.get("graded_chunks", [])
    ]

    return AskResponse(
        request_id=request_id,
        question=body.question,
        answer=result.answer,
        confidence=result.confidence,
        avg_relevance_score=result.avg_relevance_score,
        web_triggered=result.web_triggered,
        web_query=result.web_query,
        sources_used=result.sources_used,
        crag_trace=crag_trace,
        latency_ms=round((time.time() - t0) * 1000),
    )


@router.post(
    "/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED
)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a medical book (PDF, EPUB, TXT) and ingest it into ChromaDB.
    Idempotent: re-uploading the same file updates existing chunks via upsert.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Supported: {_SUPPORTED_EXTENSIONS}",
        )

    # Save to a temp path for processing
    tmp_path = Path(f"/tmp/{uuid.uuid4()}{ext}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        loader = DocumentLoader()
        pages = loader.load(tmp_path)

        chunker = BookChunker()
        chunks = chunker.chunk_pages(pages)

        embedder = Embedder()
        upserted = embedder.upsert(chunks)

        logger.info(
            "ingest_complete",
            filename=file.filename,
            pages=len(pages),
            chunks=len(chunks),
            upserted=upserted,
        )

        return IngestResponse(
            filename=file.filename or "unknown",
            pages_loaded=len(pages),
            chunks_created=len(chunks),
            chunks_upserted=upserted,
            collection_size=embedder.collection_size(),
        )
    except Exception as exc:
        logger.error("ingest_failed", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + ChromaDB status check."""
    settings = get_settings()
    embedder = Embedder()
    return HealthResponse(
        status="ok",
        collection_name=settings.collection_name,
        collection_size=embedder.collection_size(),
        groq_model=settings.groq_model,
    )


@router.get("/audit", response_model=AuditResponse)
async def audit(n: int = Query(default=50, ge=1, le=500)) -> AuditResponse:
    """Return the last n CRAG decision records from the audit log."""
    records = get_audit_logger().tail(n)
    return AuditResponse(records=records, total=len(records))
