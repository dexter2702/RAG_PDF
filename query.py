"""
query.py — CLI with Multi-turn Memory
======================================
Usage:
    python query.py                                    # interactive mode
    python query.py -q "What is this paper about?"    # single question (no memory)
"""

import argparse
import os
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rag import PDFRagEngine

console = Console()


def display_answer(result: dict):
    """Pretty-print answer and sources."""
    console.print()
    console.print(Panel(
        Text(result["answer"], style="white"),
        title="[bold cyan]Answer[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    sources = result["sources"]
    if sources:
        table = Table(title="Sources used", show_lines=True, border_style="dim")
        table.add_column("File",    style="green",  no_wrap=True, max_width=30)
        table.add_column("Page",    style="yellow", justify="center", max_width=6)
        table.add_column("Excerpt", style="dim",    max_width=80)
        for doc in sources:
            meta    = doc.metadata
            source  = os.path.basename(meta.get("source", "unknown"))
            page    = str(int(meta.get("page", 0)) + 1)
            excerpt = doc.page_content[:180].replace("\n", " ").strip() + "..."
            table.add_row(source, page, excerpt)
        console.print(table)
    console.print()


def interactive_mode(engine: PDFRagEngine):
    """
    Multi-turn interactive loop.
    Keeps a running chat_history list and passes it to every ask() call.
    History grows with each turn so the engine always has full context.
    """
    stats = engine.get_collection_stats()
    console.print(Panel(
        f"[green]Vector store loaded:[/green] {stats['total_chunks']} chunks indexed\n"
        f"[dim]Memory is ON — the bot remembers your previous questions.\n"
        f"Type 'history' to see the conversation so far.\n"
        f"Type 'clear' to reset memory. Type 'exit' to quit.[/dim]",
        title="[bold]PDF RAG — Multi-turn Mode[/bold]",
        border_style="green",
    ))

    # This list accumulates HumanMessage and AIMessage objects across turns.
    chat_history = []

    while True:
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        question = input().strip()

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if question.lower() == "clear":
            chat_history = []
            console.print("[yellow]Memory cleared.[/yellow]")
            continue

        if question.lower() == "history":
            if not chat_history:
                console.print("[dim]No history yet.[/dim]")
            else:
                for msg in chat_history:
                    role = "You" if isinstance(msg, HumanMessage) else "Bot"
                    style = "cyan" if role == "You" else "green"
                    console.print(f"[{style}]{role}:[/{style}] {msg.content[:120]}")
            continue

        if not question:
            continue

        console.print("[dim]Thinking...[/dim]")

        # Pass the current history so the engine can contextualize the question
        result = engine.ask(question, chat_history=chat_history)
        display_answer(result)

        # Append this turn to history for the next question
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["answer"]))


def single_question_mode(engine: PDFRagEngine, question: str):
    """Single question — no history, just answer and exit."""
    console.print(f"[bold]Q:[/bold] {question}")
    console.print("[dim]Searching...[/dim]")
    result = engine.ask(question, chat_history=[])
    display_answer(result)


def main():
    parser = argparse.ArgumentParser(description="Query your PDF RAG system")
    parser.add_argument("-q", "--question",  default=None,             help="Single question")
    parser.add_argument("--persist_dir",     default="./chroma_db",    help="ChromaDB directory")
    parser.add_argument("--model",           default="llama-3.1-8b-instant", help="Groq model")
    parser.add_argument("--top_k",           default=4, type=int,      help="Chunks to retrieve")
    args = parser.parse_args()

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
    main()
