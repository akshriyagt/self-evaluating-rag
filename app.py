"""
Streamlit UI for the self-evaluating RAG pipeline.

Run with:
    streamlit run app.py
"""

import streamlit as st
from datetime import datetime
import os
from src.rag_pipeline import RAGPipeline
from src import ingest

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")

st.set_page_config(page_title="Self-Evaluating RAG", layout="wide")

if "pipeline" not in st.session_state:
    with st.spinner("Loading models and vector store..."):
        st.session_state.pipeline = RAGPipeline()

if "log" not in st.session_state:
    st.session_state.log = []

st.title("Self-Evaluating RAG Pipeline")
st.caption(
    "A RAG system that scores its own answers for grounding and flags "
    "low-confidence responses instead of hallucinating."
)

with st.sidebar:
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        os.makedirs(DOCS_DIR, exist_ok=True)
        for uf in uploaded_files:
            save_path = os.path.join(DOCS_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s). Click 'Re-index documents' below.")

    existing = [
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".txt", ".md", ".pdf"))
    ] if os.path.isdir(DOCS_DIR) else []
    if existing:
        st.caption("Currently in sample_docs/:")
        for f in existing:
            st.caption(f"• {f}")

    if st.button("Re-index documents"):
        with st.spinner("Re-embedding all documents into the vector store..."):
            ingest.main()
            st.session_state.pipeline = RAGPipeline()
        st.success("Re-indexed. You can ask questions now.")

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Ask a question about the documents:")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and query.strip():
        # Build a short chat-history string from the last 3 exchanges so
        # follow-up questions ("what about remote workers?") work naturally.
        history_lines = []
        for entry in reversed(st.session_state.log[-3:]):
            history_lines.append(f"Q: {entry['query']}\nA: {entry['answer']}")
        chat_history = "\n\n".join(history_lines)

        with st.spinner("Retrieving, generating, and judging..."):
            result = st.session_state.pipeline.answer_query(query, chat_history)

        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log.insert(0, result)

        if result["flagged_uncertain"]:
            st.warning("⚠️ Low confidence / possibly ungrounded answer")
        else:
            st.success("✅ Grounded answer")

        st.markdown("### Answer")
        st.write(result["answer"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{result['confidence']:.0f}/100")
        m2.metric("Grounded", "Yes" if result["grounded"] else "No")
        m3.metric("Sources used", len(set(result.get("sources", []))))

        with st.expander("Judge reasoning"):
            st.write(result["reasoning"])

        with st.expander("Retrieved chunks (what the answer was based on)"):
            for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                st.markdown(f"**Chunk {i+1}** (source: {result['sources'][i]})")
                st.text(chunk)

with col2:
    st.markdown("### Query Log (Monitoring Dashboard)")
    if not st.session_state.log:
        st.caption("No queries yet.")
    for entry in st.session_state.log[:10]:
        icon = "⚠️" if entry["flagged_uncertain"] else "✅"
        st.markdown(f"{icon} **{entry['query']}**")
        st.caption(f"{entry['timestamp']} · confidence {entry['confidence']:.0f}")
        st.divider()