# PDF RAG System 📄🤖

A Retrieval-Augmented Generation (RAG) system that lets you ask questions
about your PDF documents using free, local embeddings and Groq LLaMA 3.1.

## Tech Stack
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (free, runs locally)
- **Vector Store**: FAISS
- **LLM**: Groq LLaMA 3.1 (free API)
- **Orchestration**: LangChain (LCEL)

## Setup

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pdf-rag.git
cd pdf-rag

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your API key
cp .env.example .env
# Open .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com

### 5. Add PDFs and ingest
mkdir docs
# Copy your PDF files into docs/
python ingest.py

### 6. Ask questions!
python query.py -q "What is this document about?"
# Or interactive mode:
python query.py