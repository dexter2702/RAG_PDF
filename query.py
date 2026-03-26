"""
query.py — Interactive Q&A CLI
================================
After running ingest.py, use this to ask questions about your PDFs.

Usage:
    python query.py                         # interactive mode (keep asking)
    python query.py -q "What is this PDF about?"   # single question mode
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rag import PDFRagEngine

console = Console()


def display_answer(question: str, result: dict):
    """Pretty-print the answer and its source chunks."""

    # ── Answer panel ──────────────────────────────────────
    console.print()
    console.print(Panel(
        Text(result["answer"], style="white"),
        title=f"[bold cyan]Answer[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Sources table ──────────────────────────────────────
    sources = result["sources"]
    if sources:
        table = Table(title="Sources used", show_lines=True, border_style="dim")
        table.add_column("File",    style="green",  no_wrap=True, max_width=30)
        table.add_column("Page",    style="yellow", justify="center", max_width=6)
        table.add_column("Excerpt", style="dim",    max_width=80)

        for doc in sources:
            meta    = doc.metadata
            source  = os.path.basename(meta.get("source", "unknown"))
            page    = str(meta.get("page", "?") + 1)  # 0-indexed → 1-indexed
            excerpt = doc.page_content[:180].replace("\n", " ").strip() + "..."
            table.add_row(source, page, excerpt)

        console.print(table)
    console.print()


def interactive_mode(engine: PDFRagEngine):
    """Run a continuous Q&A loop until the user types 'exit'."""
    stats = engine.get_collection_stats()
    console.print(Panel(
        f"[green]Vector store loaded:[/green] {stats['total_chunks']} chunks indexed\n"
        f"[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]",
        title="[bold]PDF RAG — Ready[/bold]",
        border_style="green",
    ))

    while True:
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        question = input().strip()

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not question:
            continue

        console.print("[dim]Searching and generating answer...[/dim]")
        result = engine.ask(question)
        display_answer(question, result)


def single_question_mode(engine: PDFRagEngine, question: str):
    """Answer a single question and exit."""
    console.print(f"[bold]Q:[/bold] {question}")
    console.print("[dim]Searching...[/dim]")
    result = engine.ask(question)
    display_answer(question, result)


def main():
    import os  # needed inside display_answer

    parser = argparse.ArgumentParser(description="Query your PDF RAG system")
    parser.add_argument("-q", "--question",    default=None,          help="Single question to ask")
    parser.add_argument("--persist_dir",       default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--model",             default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--top_k",             default=4, type=int,   help="Chunks to retrieve")
    args = parser.parse_args()

    # Load the RAG engine (loads vector store from disk)
    try:
        engine = PDFRagEngine(
            persist_dir=args.persist_dir,
            model_name=args.model,
            top_k=args.top_k,
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    if args.question:
        single_question_mode(engine, args.question)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    import os
    main()
