"""
rag.py — Core RAG Engine (Multi-turn memory version)
=====================================================
Key upgrade: Two-step chain
  Step 1 — Contextualize the question using chat history
           "What did they study?" → "What did Hinton study?"
  Step 2 — Retrieve relevant chunks using the rewritten question
           then answer using history + context + question
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# Step 1 prompt — rewrites vague follow-up questions into standalone ones.
# The LLM reads the history and figures out what "they", "it", "that" refers to.
CONTEXTUALIZE_PROMPT = """Given the conversation history and the latest user \
question, reformulate the question as a standalone question that makes sense \
without the history. Do NOT answer the question — just rewrite it if needed. \
If it's already standalone, return it exactly as is."""

# Step 2 prompt — answers using retrieved context + full conversation history.
# MessagesPlaceholder injects the actual chat history messages here.
ANSWER_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided PDF documents.

Rules:
1. Only use information from the CONTEXT below to answer.
2. If the answer is not in the context, say "I don't have enough information \
in the provided documents to answer this."
3. Be concise and direct.
4. You may use the conversation history to give a more coherent answer.

CONTEXT:
{context}"""


def format_docs(docs: list) -> str:
    """Join retrieved chunks into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_history(raw_history: list) -> list:
    """
    Convert a list of plain dicts into LangChain message objects.

    Accepts two formats:
      - {"role": "user"/"assistant", "content": "..."}  ← from Streamlit
      - Already HumanMessage / AIMessage objects         ← from CLI

    Returns a list of HumanMessage / AIMessage objects.
    """
    messages = []
    for item in raw_history:
        if isinstance(item, (HumanMessage, AIMessage)):
            messages.append(item)
        elif isinstance(item, dict):
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
    return messages


class PDFRagEngine:
    """
    Multi-turn RAG engine.

    Usage:
        engine = PDFRagEngine()

        # First question — no history yet
        result = engine.ask("What is this paper about?", chat_history=[])

        # Follow-up — pass the accumulated history
        result = engine.ask("Who are the authors?", chat_history=[...])
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = "llama-3.1-8b-instant",
        top_k: int = 4,
    ):
        self.persist_dir = persist_dir
        self.top_k = top_k

        # Embedding model — same one used in ingest.py (must match!)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # LLM
        self.llm = ChatGroq(model=model_name, temperature=0)

        # Vector store
        self.vectorstore = self._load_vectorstore()

        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        # Build both chains
        self.contextualize_chain = self._build_contextualize_chain()
        self.chain = self._build_rag_chain()

    def _load_vectorstore(self) -> FAISS:
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Vector store not found at '{self.persist_dir}'.\n"
                "Run ingest.py first to index your PDF files."
            )
        return FAISS.load_local(
            self.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _build_contextualize_chain(self):
        """
        Chain that rewrites a follow-up question into a standalone question.

        Input:  {"input": question, "chat_history": [HumanMessage, AIMessage, ...]}
        Output: rewritten question string
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),   # injects history here
            ("human", "{input}"),
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_rag_chain(self):
        """
        Full RAG chain with memory.

        Flow:
          input + history
              │
              ├─► contextualize_question() ─► retriever ─► format_docs ─► {context}
              │
              └─► {input} passthrough
                              │
                              ▼
                      qa_prompt (system + history + human)
                              │
                              ▼
                             LLM
                              │
                              ▼
                           answer string
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("chat_history"),   # full history injected here
            ("human", "{input}"),
        ])

        def contextualized_question(input_dict: dict):
            """
            If there's chat history, rewrite the question.
            If it's the first question, use it directly — no need to call LLM.
            This saves one LLM call on every first question.
            """
            if input_dict.get("chat_history"):
                return self.contextualize_chain
            # No history → return the question string directly
            return input_dict["input"]

        chain = (
            RunnablePassthrough.assign(
                # Retrieve using the (possibly rewritten) question
                context=contextualized_question | self.retriever | format_docs
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def ask(self, question: str, chat_history: list = None) -> dict:
        """
        Ask a question with optional chat history.

        Args:
            question    : The user's current question.
            chat_history: List of past messages. Accepts either:
                          - [{"role": "user"/"assistant", "content": "..."}]  (Streamlit)
                          - [HumanMessage(...), AIMessage(...)]               (CLI)
                          Pass [] or None for the first question.

        Returns:
            {
                "answer"  : str,
                "sources" : list of Document objects
            }
        """
        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        # Normalise history into LangChain message objects
        history = build_history(chat_history or [])

        # Retrieve source docs using the rewritten question
        # (We rewrite manually here so we can return the right source docs)
        if history:
            rewritten = self.contextualize_chain.invoke({
                "input": question,
                "chat_history": history,
            })
        else:
            rewritten = question

        source_docs = self.retriever.invoke(rewritten)

        # Run the full chain for the answer
        answer = self.chain.invoke({
            "input": question,
            "chat_history": history,
        })

        return {
            "answer": answer,
            "sources": source_docs,
        }

    def get_collection_stats(self) -> dict:
        count = self.vectorstore.index.ntotal
        return {"total_chunks": count, "persist_dir": self.persist_dir}
