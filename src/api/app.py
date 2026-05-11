"""
src/api/app.py

FastAPI application factory.
- Lifespan context manager for startup/shutdown logging
- CORS configured for local dev (Streamlit removed, plain HTML frontend)
- Static file serving for frontend/index.html
- All routes mounted under /api/v1
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve the frontend directory relative to this file
# Expects: frontend/index.html at the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "startup",
        groq_model=settings.groq_model,
        collection=settings.collection_name,
        chroma_path=settings.chroma_path,
        frontend=str(_FRONTEND_DIR),
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
        allow_origins=["*"],          # frontend is served from same origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes ──────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Static frontend ─────────────────────────────────────────────────────
    if _FRONTEND_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_FRONTEND_DIR)),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        async def serve_frontend() -> FileResponse:
            return FileResponse(str(_FRONTEND_DIR / "index.html"))
    else:
        logger.warning("frontend_dir_missing", path=str(_FRONTEND_DIR))

    return app


app = create_app()