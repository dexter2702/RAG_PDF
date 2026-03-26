"""
rag.py — Core RAG Engine (Modern LCEL version)
===============================================
Uses LangChain Expression Language (LCEL) instead of the
deprecated RetrievalQA chain. No langchain.chains needed.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the
provided PDF documents. Follow these rules:

1. Only use information from the CONTEXT below to answer.
2. If the answer is not in the context, say "I don't have enough information
   in the provided documents to answer this."
3. Be concise and direct. Cite which part of the document supports your answer
   when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def format_docs(docs):
    """Joins retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


class PDFRagEngine:

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = "gpt-3.5-turbo",
        top_k: int = 4,
    ):
        self.persist_dir = persist_dir
        self.top_k = top_k

        # 1. Embedding model (local, free)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. LLM for answer generation
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        # 3. Load vector store from disk
        self.vectorstore = self._load_vectorstore()

        # 4. Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # 5. Build the LCEL chain
        self.chain = self._build_chain()

    def _load_vectorstore(self):
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Vector store not found at '{self.persist_dir}'.\n"
                "Run ingest.py first."
            )
        return FAISS.load_local(
            self.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _build_chain(self):
        """
        Builds the chain using LCEL (LangChain Expression Language).

        Flow:
          question -> retriever -> format_docs -> prompt -> llm -> answer
        """
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        chain = (
            {
                "context": self.retriever | format_docs,  # retrieve + join chunks
                "question": RunnablePassthrough(),         # pass question through as-is
            }
            | prompt             # fill the prompt template
            | self.llm           # send to LLM
            | StrOutputParser()  # extract plain string from LLM response
        )
        return chain

    def ask(self, question: str) -> dict:
        """
        Ask a question. Returns:
          - "answer"  : the LLM's answer string
          - "sources" : list of source chunks used
        """
        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        # Get source docs separately so we can show them
        source_docs = self.retriever.invoke(question)

        # Run the chain for the answer
        answer = self.chain.invoke(question)

        return {
            "answer": answer,
            "sources": source_docs,
        }

    def get_collection_stats(self) -> dict:
        """Returns basic stats about the vector store."""
        count = self.vectorstore.index.ntotal  # FAISS-specific
        return {"total_chunks": count, "persist_dir": self.persist_dir}
