"""
src/api/app.py

FastAPI application factory.
- Lifespan context manager for startup/shutdown logging
- CORS configured for local Streamlit dev
- All routes mounted under /api/v1
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "startup",
        groq_model=settings.groq_model,
        collection=settings.collection_name,
        chroma_path=settings.chroma_path,
    )
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Medical CRAG API",
        description=(
            "Corrective RAG pipeline for medical Q&A. "
            "Powered by Groq (llama-3.3-70b-versatile) + ChromaDB + Tavily."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
