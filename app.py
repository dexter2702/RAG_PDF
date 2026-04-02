"""
app.py — Streamlit Chat UI with Multi-turn Memory
==================================================
Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from rag import PDFRagEngine

load_dotenv()

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="centered")

PERSIST_DIR   = "./chroma_db"
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── Cached heavy objects (model weights — truly load once) ─────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "engine"         not in st.session_state:
    # Try to load an existing vector store on first run
    if os.path.exists(PERSIST_DIR):
        try:
            st.session_state.engine = PDFRagEngine(persist_dir=PERSIST_DIR)
        except Exception:
            st.session_state.engine = None
    else:
        st.session_state.engine = None

# ── Helper: reload engine from disk into session state ────────────────────────
def reload_engine():
    """Create a fresh PDFRagEngine and store it in session state."""
    st.session_state.engine = PDFRagEngine(persist_dir=PERSIST_DIR)

# ── Helper: wipe everything and start fresh ───────────────────────────────────
def clear_all():
    """Delete the vector store from disk and reset all session state."""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)          # delete chroma_db/ folder entirely
    st.session_state.engine         = None
    st.session_state.messages       = []
    st.session_state.uploaded_files = []

# ── PDF ingestion ──────────────────────────────────────────────────────────────
def ingest_pdf(uploaded_file, embeddings):
    """Load, chunk, embed and save a PDF. Merges into existing store if present."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Merge into existing store OR create new one
    if os.path.exists(PERSIST_DIR):
        existing = FAISS.load_local(
            PERSIST_DIR, embeddings, allow_dangerous_deserialization=True
        )
        existing.add_documents(chunks)
        existing.save_local(PERSIST_DIR)
    else:
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(PERSIST_DIR)

    os.unlink(tmp_path)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chat")
    st.caption("Upload PDFs, then ask anything about them.")
    st.divider()

    # ── Upload ──
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader(
        label="Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        embeddings = load_embeddings()
        any_new = False
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Ingesting {file.name}..."):
                    try:
                        ingest_pdf(file, embeddings)
                        st.session_state.uploaded_files.append(file.name)
                        st.success(f"✓ {file.name}")
                        any_new = True
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # Reload engine ONCE after all new files are processed
        if any_new:
            reload_engine()

    # ── Indexed files list ──
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Indexed files")
        for name in st.session_state.uploaded_files:
            st.markdown(f"📎 {name}")

    # ── Chat history preview ──
    if st.session_state.messages:
        st.divider()
        st.subheader("Chat history")
        for msg in st.session_state.messages:
            icon = "🧑" if msg["role"] == "user" else "🤖"
            preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            st.markdown(f"{icon} {preview}")

    # ── Action buttons ──
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.messages:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    with col2:
        # This is the key fix — lets user wipe old documents and start fresh
        if st.button("🔄 New session", use_container_width=True, type="primary"):
            clear_all()
            st.rerun()

    if st.session_state.uploaded_files:
        st.caption("⚠️ 'New session' removes all indexed documents.")

# ── Main chat area ─────────────────────────────────────────────────────────────
st.header("Chat with your PDFs")

if st.session_state.engine is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask something about your PDFs..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = st.session_state.messages[:-1]
                result  = st.session_state.engine.ask(question, chat_history=history)
                answer  = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 {len(sources)} source chunks used"):
                        for i, doc in enumerate(sources, 1):
                            fname = os.path.basename(doc.metadata.get("source", "unknown"))
                            page  = doc.metadata.get("page", "?")
                            st.markdown(f"**Chunk {i}** — `{fname}` page {int(page)+1}")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
