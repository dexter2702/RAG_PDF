# PDF RAG System 📄🤖

A Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents
and have a multi-turn conversation about them — completely free, no OpenAI credits needed.

## Features
- 📄 Upload PDFs directly from the browser
- 💬 Multi-turn memory — the bot remembers your previous questions
- 🔍 Semantic search over your documents using FAISS
- 🆓 Free local embeddings (HuggingFace) + free LLM (Groq LLaMA 3.1)
- 📚 Shows source chunks used for every answer
- 🔄 New session button to switch documents cleanly
- 💻 CLI mode also available with full memory support

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector Store | FAISS |
| LLM | Groq LLaMA 3.1 (free API) |
| Orchestration | LangChain LCEL |
| Frontend | Streamlit |

## Project Structure

```
pdf_rag/
├── app.py              ← Streamlit web UI
├── rag.py              ← Core RAG engine (multi-turn memory)
├── ingest.py           ← CLI PDF loader and indexer
├── query.py            ← CLI question answering
├── requirements.txt
├── .env.example
└── docs/               ← Drop PDFs here for CLI ingestion
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag.git
cd pdf-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
```bash
cp .env.example .env
# Open .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
Upload a PDF from the sidebar and start chatting!

### 6. Or use the CLI
```bash
# Index your PDFs first
python ingest.py

# Interactive mode (with memory)
python query.py

# Single question
python query.py -q "What is this paper about?"
```

## How It Works

**Indexing (one-time per document):**
```
PDF → split into chunks → embed each chunk → store in FAISS vector store
```

**Query (every question):**
```
Question + chat history → rewrite as standalone question
        → search FAISS → retrieve top-4 chunks
        → build prompt → Groq LLaMA 3.1 → answer
```

## Roadmap
- [x] Basic RAG pipeline
- [x] Streamlit web UI with PDF upload
- [x] Multi-turn conversation memory
- [ ] Hybrid search (BM25 + vector)
- [ ] Metadata filtering by document
- [ ] Adaptive RAG with LangGraph
