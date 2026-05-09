# Medical Q&A — Corrective RAG System

Production-grade Corrective RAG pipeline for medical literature Q&A,
built with LangGraph, Groq (llama-3.3-70b-versatile), ChromaDB, and Tavily.

## Architecture

```
User Query
    │
    ▼
Vector Retrieval (ChromaDB)
    │
    ▼
Relevance Grader (Groq LLM) ──→ scores each chunk 0.0–1.0
    │
    ▼
Confidence Gate
    ├── HIGH  (≥0.7)  ──→ Use local docs directly
    ├── LOW   (<0.4)  ──→ Rewrite query + Tavily web search
    └── AMBIG (0.4–0.7) ──→ Fuse local docs + web results
    │
    ▼
Answer Generation (Groq LLM) with citations + confidence badge
    │
    ▼
Structured Response (answer, sources, confidence, crag_trace)
```

## Project Structure

```
medical_crag/
├── src/
│   ├── ingestion/          # Book chunking & embedding pipeline
│   │   ├── __init__.py
│   │   ├── document_loader.py   # PDF/EPUB/TXT loader
│   │   ├── chunker.py           # Semantic chunking strategy
│   │   └── embedder.py          # Embedding + ChromaDB upsert
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py         # ChromaDB similarity search
│   ├── grading/
│   │   ├── __init__.py
│   │   └── relevance_grader.py  # LLM-based chunk scoring
│   ├── generation/
│   │   ├── __init__.py
│   │   └── answer_generator.py  # Final answer synthesis
│   ├── web_search/
│   │   ├── __init__.py
│   │   └── tavily_search.py     # Tavily fallback + query rewriter
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI application
│   │   ├── routes.py            # /ask, /ingest, /health endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Pydantic settings (env-driven)
│       ├── logger.py            # Structured logging
│       └── audit_logger.py      # CRAG decision audit trail
├── graph/
│   ├── __init__.py
│   ├── crag_graph.py            # LangGraph state machine
│   └── state.py                 # TypedDict graph state
├── scripts/
│   └── ingest_book.py           # CLI: python scripts/ingest_book.py <path>
├── tests/
│   ├── test_grader.py
│   ├── test_retriever.py
│   └── test_graph.py
├── configs/
│   └── prompts.py               # All LLM prompts (versioned)
├── .env.example
├── requirements.txt
├── Makefile
└── streamlit_app.py             # Demo UI
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Copy env and fill in keys
cp .env.example .env

# 3. Ingest your book
python scripts/ingest_book.py data/raw/your_book.pdf

# 4. Start the API
uvicorn src.api.app:app --reload

# 5. Launch the Streamlit demo
streamlit run streamlit_app.py
```

## Environment Variables

| Variable          | Description                        |
|-------------------|------------------------------------|
| `GROQ_API_KEY`    | Groq API key                       |
| `TAVILY_API_KEY`  | Tavily search API key              |
| `CHROMA_PATH`     | ChromaDB persist directory         |
| `COLLECTION_NAME` | ChromaDB collection name           |
| `TOP_K`           | Number of chunks to retrieve       |
| `HIGH_CONF_THRESHOLD` | Float ≥ this → use local docs  |
| `LOW_CONF_THRESHOLD`  | Float < this → web fallback    |
| `LOG_LEVEL`       | DEBUG / INFO / WARNING             |

## API Endpoints

| Method | Path        | Description                        |
|--------|-------------|------------------------------------|
| POST   | `/ask`      | Ask a medical question             |
| POST   | `/ingest`   | Ingest a document via upload       |
| GET    | `/health`   | Liveness + ChromaDB status         |
| GET    | `/audit`    | Last N CRAG decision logs          |

## Demo Questions (for recruiters)

```bash
# High confidence — answered from book
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the mechanism of action of metformin?"}'

# Low confidence — triggers Tavily fallback
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Latest 2024 guidelines for COVID booster in immunocompromised adults?"}'
```
