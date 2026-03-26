"""
ingest.py — PDF Loader & Indexer
=================================
Run this ONCE (or whenever you add new PDFs).
It will:
  1. Load all PDFs from your ./docs folder
  2. Split them into overlapping chunks
  3. Embed each chunk using OpenAI
  4. Save everything to ChromaDB on disk

Usage:
    python ingest.py
    python ingest.py --docs_dir ./my_papers --chunk_size 800
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()


def load_pdfs(docs_dir: str) -> list:
    """
    Load all PDFs from a directory.
    Each page of a PDF becomes one LangChain 'Document' object
    with metadata: {"source": "filename.pdf", "page": 0}
    """
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        console.print(f"[yellow]Created '{docs_dir}' folder. Add your PDFs there and re-run.[/yellow]")
        return []

    # DirectoryLoader finds all .pdf files recursively
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",           # match all PDFs, including subfolders
        loader_cls=PyPDFLoader,    # use PyPDF to read each file
        show_progress=True,
    )

    documents = loader.load()
    console.print(f"[green]Loaded {len(documents)} pages from PDFs in '{docs_dir}'[/green]")
    return documents


def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Split long documents into smaller overlapping chunks.

    Why chunking?
      LLMs have a limited context window. A 50-page PDF won't fit in one prompt.
      We split it into ~500-character pieces so only the RELEVANT pieces are
      sent to the LLM.

    Why overlap?
      If a key sentence sits at the boundary between two chunks, overlap ensures
      it appears in at least one chunk fully. 50-100 char overlap is usually enough.

    RecursiveCharacterTextSplitter tries to split at:
      paragraph breaks → newlines → sentences → words → characters
    (in that order) to keep semantic units together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,           # count raw characters (not tokens)
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    console.print(f"[green]Split into {len(chunks)} chunks "
                  f"(size={chunk_size}, overlap={chunk_overlap})[/green]")
    return chunks


def build_vectorstore(chunks, persist_dir):
    # Free model, downloads once (~90MB), runs locally after that
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    console.print(f"[cyan]Embedding {len(chunks)} chunks locally (no API needed)...[/cyan]")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    console.print(f"[bold green]✓ Vector store saved to '{persist_dir}'[/bold green]")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--docs_dir",      default="./docs",       help="Folder with PDFs")
    parser.add_argument("--persist_dir",   default="./chroma_db",  help="Where to save vectors")
    parser.add_argument("--chunk_size",    default=600, type=int,   help="Chars per chunk")
    parser.add_argument("--chunk_overlap", default=80,  type=int,   help="Overlap between chunks")
    args = parser.parse_args()

    console.rule("[bold]PDF RAG — Ingestion Pipeline[/bold]")

    # Step 1: Load
    documents = load_pdfs(args.docs_dir)
    if not documents:
        return

    # Step 2: Split
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    # Step 3: Embed + Store
    build_vectorstore(chunks, args.persist_dir)

    console.rule("[bold green]Ingestion complete! Run query.py to start asking questions.[/bold green]")


if __name__ == "__main__":
    main()
