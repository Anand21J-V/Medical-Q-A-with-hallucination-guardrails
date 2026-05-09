"""
streamlit_app.py

Recruiter-ready demo UI for the Medical CRAG system.

Panels:
  1. Answer + confidence badge + source list
  2. CRAG trace table (chunk scores, per-chunk decision)
  3. Session audit log (correction rate over session)

Run: streamlit run streamlit_app.py
"""

import json
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

# ── Confidence badge colours ──────────────────────────────────────────────────
CONF_COLOUR = {
    "high": "#1D9E75",
    "ambiguous": "#BA7517",
    "low": "#D85A30",
}
CONF_LABEL = {
    "high": "HIGH — answered from local book",
    "ambiguous": "AMBIGUOUS — fused local + web",
    "low": "LOW — web search triggered",
}

# ── Demo questions ────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What is the mechanism of action of metformin in type 2 diabetes?",
    "Describe the pathophysiology of myocardial infarction.",
    "What are the first-line antibiotics for community-acquired pneumonia?",
    "Latest 2024 recommendations for COVID-19 booster in immunocompromised adults?",
    "What are the diagnostic criteria for sepsis?",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical CRAG",
    page_icon="🏥",
    layout="wide",
)

st.title("Medical Q&A — Corrective RAG")
st.caption(
    "Groq llama-3.3-70b-versatile · ChromaDB · Tavily · LangGraph"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "web_trigger_count" not in st.session_state:
    st.session_state.web_trigger_count = 0


# ── Sidebar: ingest + stats ───────────────────────────────────────────────────
    # ── Sidebar: stats only (book is pre-ingested, no upload needed) ─────────────
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.info("Medical book is pre-loaded on the backend. No upload needed.")
    
    st.divider()
    st.header("Session stats")
    
    total_q = len(st.session_state.history)
    web_count = st.session_state.web_trigger_count
    correction_rate = (
        f"{round(web_count / total_q * 100)}%" if total_q > 0 else "—"
    )

    col1, col2 = st.columns(2)
    col1.metric("Questions asked", total_q)
    col2.metric("Web fallback rate", correction_rate)

    if total_q > 0:
        high = sum(1 for h in st.session_state.history if h["confidence"] == "high")
        amb = sum(1 for h in st.session_state.history if h["confidence"] == "ambiguous")
        low = sum(1 for h in st.session_state.history if h["confidence"] == "low")
        st.caption(f"HIGH {high} · AMBIGUOUS {amb} · LOW {low}")

    st.divider()

    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API online · {health['collection_size']} chunks indexed")
    except Exception:
        st.warning("API offline — start with: uvicorn src.api.app:app --reload")


# ── Main input ────────────────────────────────────────────────────────────────
st.subheader("Ask a medical question")

with st.form("question_form"):
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the mechanism of action of metformin?",
    )
    col_a, col_b = st.columns([1, 5])
    submitted = col_a.form_submit_button("Ask", type="primary")
    top_k = col_b.select_slider("Chunks to retrieve", options=[3, 5, 7, 10], value=5)

st.caption("Or try a demo question:")
demo_cols = st.columns(len(DEMO_QUESTIONS))
for i, dq in enumerate(DEMO_QUESTIONS):
    if demo_cols[i].button(dq[:35] + "…", key=f"demo_{i}"):
        question = dq
        submitted = True

# ── Query execution ───────────────────────────────────────────────────────────
if submitted and question.strip():
    with st.spinner("Running CRAG pipeline..."):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/ask",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Start it with: `uvicorn src.api.app:app --reload`")
            st.stop()
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

    # Record in history
    st.session_state.history.append(data)
    if data["web_triggered"]:
        st.session_state.web_trigger_count += 1

    # ── Answer panel ──────────────────────────────────────────────────────
    conf = data["confidence"]
    colour = CONF_COLOUR.get(conf, "#888")
    label = CONF_LABEL.get(conf, conf)

    st.markdown(f"""
    <div style="border-left: 4px solid {colour}; padding: 10px 16px;
                background: #f9f9f9; border-radius: 4px; margin-bottom: 12px;">
        <span style="color:{colour}; font-weight:600; font-size:13px;">
            ● {label.upper()}
        </span>
        <span style="color:#888; font-size:12px; margin-left:12px;">
            avg score: {data['avg_relevance_score']:.2f} · {data['latency_ms']}ms
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Answer")
    st.write(data["answer"])

    if data["web_triggered"]:
        st.info(f"Web search triggered — query used: *\"{data['web_query']}\"*")

    with st.expander("Sources used"):
        for src in data["sources_used"]:
            st.markdown(f"- {src}")

    # ── CRAG trace table ──────────────────────────────────────────────────
    st.markdown("#### CRAG trace")
    st.caption("Per-chunk relevance scores that drove the confidence decision")

    if data["crag_trace"]:
        trace_data = []
        for t in data["crag_trace"]:
            decision = (
                "USED" if t["relevance_score"] >= 0.7
                else "PARTIAL" if t["relevance_score"] >= 0.4
                else "DISCARDED"
            )
            trace_data.append({
                "Source": t["source"],
                "Page": t["page_number"],
                "Score": round(t["relevance_score"], 3),
                "Decision": decision,
                "Preview": t["text_preview"][:80] + "…",
            })
        st.dataframe(trace_data, use_container_width=True)
    else:
        st.caption("No local chunks retrieved (full web search path)")


# ── Audit log panel ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander("Session audit log", expanded=False):
        for record in reversed(st.session_state.history[-10:]):
            icon = "🟢" if record["confidence"] == "high" else (
                "🟡" if record["confidence"] == "ambiguous" else "🔴"
            )
            st.markdown(
                f"{icon} **{record['confidence'].upper()}** | "
                f"score={record['avg_relevance_score']:.2f} | "
                f"web={'yes' if record['web_triggered'] else 'no'} | "
                f"{record['question'][:80]}"
            )
