from __future__ import annotations
import os
import streamlit as st
from advanced_rag import (
    CHAT_MODEL, EMBED_MODEL, SUPPORTED_EXTENSIONS, TOP_K,
    delete_all_documents, delete_document, list_documents, process_document, query_document,
)

st.set_page_config(page_title="AI Mastery — Advanced RAG", page_icon="🧠", layout="wide")
st.title("🧠 AI Mastery — Advanced RAG")
st.caption("Multi-format upload • SQLite • ChromaDB • BM25 • RRF • grounded generation")

with st.sidebar:
    st.header("⚙️ Settings")
    if os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        st.success("OpenAI API key detected")
    else:
        st.error("OPENAI_API_KEY is missing. Add it to .env and restart.")
    top_k = st.slider("Hybrid Top-K", 1, 10, TOP_K)
    st.divider()
    st.subheader("Architecture")
    st.code("""PDF / DOCX / CSV / TXT / MD
 ↓
Extract + Chunk
 ├──→ SQLite records
 └──→ ChromaDB vectors
          +
        BM25
          ↓
      Query Router
      /         \\
 SQL/filter   Hybrid
      \\         /
        Evidence
           ↓
          LLM""")
    st.caption(f"Chat: `{CHAT_MODEL}`")
    st.caption(f"Embeddings: `{EMBED_MODEL}`")
    st.caption(f"Supported files: {', '.join(SUPPORTED_EXTENSIONS)}")
    st.divider()
    if list_documents() and st.button("🧹 Delete ALL documents", use_container_width=True):
        delete_all_documents()
        st.rerun()

st.subheader("📄 1. Upload Knowledge Base")
st.caption("You can upload several files at once, in any mix of the supported formats.")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True,
)

if uploaded_files and st.button("🚀 Process Document(s)", type="primary", use_container_width=True):
    successes, failures = [], []
    progress = st.progress(0.0, text="Starting...")
    for i, uploaded in enumerate(uploaded_files, 1):
        progress.progress(i / len(uploaded_files), text=f"Processing {uploaded.name}...")
        try:
            info = process_document(uploaded.getvalue(), uploaded.name)
            successes.append(info)
        except Exception as exc:
            failures.append((uploaded.name, str(exc)))
    progress.empty()
    for info in successes:
        st.success(f"Processed `{info['filename']}` ({info['file_type']}) — {info['units']} units, {info['chunks']} chunks, {info['records']} structured records.")
    for name, error in failures:
        st.error(f"Failed to process `{name}`: {error}")

documents = list_documents()
st.divider()
st.subheader("📚 Indexed Documents")

if documents:
    for doc in documents:
        c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
        with c1:
            st.write(f"**{doc['filename']}**")
            st.caption(doc["document_id"])
        with c2: st.metric("Type", doc["file_type"].upper())
        with c3: st.metric("Units", doc["units"])
        with c4: st.metric("Records", doc["records"])
        with c5:
            if st.button("🗑️ Delete", key=f"delete_{doc['document_id']}"):
                delete_document(doc["document_id"])
                st.rerun()
else:
    st.info("No documents indexed yet.")

if documents:
    st.divider()
    st.subheader("🔎 2. Ask a Question")

    doc_options = {"🌐 All documents": None}
    for doc in documents:
        doc_options[f"{doc['filename']} ({doc['document_id']})"] = doc["document_id"]
    scope_label = st.selectbox("Search scope", list(doc_options.keys()))
    scope_document_id = doc_options[scope_label]

    examples = [
        "List all employees in the Engineering department",
        "List all employees in the Engineering department who are on leave.",
        'Which employees work in Bengaluru and have the status "Active"?',
        'How many employees are currently "On Leave"?',
    ]
    selected = st.selectbox("Training questions", ["Custom question"] + examples)
    question = st.text_input("Your question", value="" if selected == "Custom question" else selected)
    if st.button("💬 Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Routing query and gathering verified evidence..."):
                    result = query_document(question, document_id=scope_document_id, top_k=top_k)
                st.divider()
                st.subheader("💬 Answer")
                st.markdown(result["answer"])
                st.subheader("🧠 Query Analysis")
                a, b, c = st.columns(3)
                with a: st.metric("Strategy", result["strategy"])
                with b: st.metric("Scope", result["document_name"])
                with c:
                    value = result.get("record_count", 0) if result["strategy"] != "Hybrid retrieval" else len(result.get("sources", []))
                    st.metric("Records / Chunks", value)
                with st.expander("🔬 Router Details"):
                    st.json(result.get("route", {}))
                if result.get("filters"):
                    st.write("**Applied filters:**")
                    st.json(result["filters"])
                if result.get("matched_records"):
                    st.subheader("📋 Verified Records")
                    st.dataframe(result["matched_records"], use_container_width=True, hide_index=True)
                if result.get("sources"):
                    for source in result["sources"]:
                        if source.get("type") == "live_api":
                            # live_lookup's source shape ({"type", "provider", "city"}) is
                            # deliberately different from a hybrid_retrieval chunk -- there's
                            # no filename/rank/vector_rank/bm25_rank because nothing was
                            # retrieved from storage, it was fetched live at question time.
                            st.info(f"🌐 Live API: {source.get('provider', 'external API')} — "
                                    f"city: {source.get('city', '?')} (fetched live, not stored)")
                        else:
                            with st.expander(f"{source['filename']} · Chunk {source['rank']} | Vector #{source['vector_rank']} | BM25 #{source['bm25_rank']}"):
                                st.write(source["text"])
                st.info("🎓 Structured questions use verified extracted records; other questions fall back to hybrid retrieval.")
            except Exception as exc:
                st.error(f"Query failed: {exc}")