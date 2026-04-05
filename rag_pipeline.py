"""
rag_pipeline.py
---------------
Core RAG pipeline — model-agnostic, works with all HuggingFace backends.

PIPELINE:
  User Query
    → FAISS semantic search  (bi-encoder, fast)
    → Similarity threshold filter
    → Cross-encoder re-ranking  (optional, accurate)
    → Prompt construction  (chat history + context + question)
    → HuggingFace LLM generation
    → Answer + source citations
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

from embeddings import similarity_search_with_threshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS  (one per model family)
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    question: str,
    context: str,
    chat_history: str,
    task: str = "text-generation",
    model_id: str = "",
) -> str:
    """
    Build the full prompt string for the LLM.

    Different model families need different prompt formats:
      - Flan-T5       : plain instruction (seq2seq, no chat template)
      - TinyLlama     : <|system|>/<|user|>/<|assistant|> tokens
      - Phi-2         : Instruct: / Output:
      - Phi-3         : <|user|>/<|end|>/<|assistant|>
    """
    system_msg = (
        "You are a precise document assistant. Answer ONLY from the provided "
        "context. If the context lacks the answer, say so clearly. "
        "Be concise and cite the source document/page when possible."
    )

    core = (
        f"CONVERSATION HISTORY:\n{chat_history}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    # Seq2Seq models (Flan-T5) — plain text
    if task == "text2text-generation":
        return (
            f"{system_msg}\n\n"
            f"Chat history:\n{chat_history}\n\n"
            f"Context from documents:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    # TinyLlama chat format
    if "tinyllama" in model_id.lower():
        return (
            f"<|system|>\n{system_msg}<|end|>\n"
            f"<|user|>\n{core}<|end|>\n"
            f"<|assistant|>\n"
        )

    # Phi-3 chat format
    if "phi-3" in model_id.lower():
        return (
            f"<|user|>\n{system_msg}\n\n{core}<|end|>\n"
            f"<|assistant|>\n"
        )

    # Phi-2 instruct format (default causal LM fallback)
    return f"Instruct: {system_msg}\n\n{core}\nOutput:"


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL RE-RANKER  (cross-encoder)
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder for higher precision.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (~22 MB, CPU-friendly)
    Trained on MS-MARCO passage ranking — returns a relevance logit score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Cross-encoder loaded: '{self.model_name}' ✓")
            except ImportError:
                logger.warning("sentence-transformers missing — skipping re-rank.")
                self._model = "unavailable"

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_n: int = 3,
    ) -> List[Tuple[Document, float]]:
        self._load()
        if self._model == "unavailable" or not docs_with_scores:
            return docs_with_scores[:top_n]

        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        ce_scores = self._model.predict(pairs)

        reranked = sorted(
            zip([d for d, _ in docs_with_scores], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info(f"Re-ranked {len(reranked)} → top {top_n}")
        return [(doc, float(s)) for doc, s in reranked[:top_n]]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════



def _format_history(chat_history: list) -> str:
    if not chat_history:
        return "None"
    lines = []
    for msg in chat_history:
        if msg["role"] == "user":
            lines.append(f"Human: {msg['content']}")
        else:
            lines.append(f"Assistant: {msg['content']}")
    return "\n".join(lines)


def _format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    if not docs_with_scores:
        return "No relevant context found."
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i} | {src} | Page {page} | Relevance {score:.2f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_answer(raw: Any) -> str:
    """Normalise LLM output to a plain string."""
    if hasattr(raw, "content"):          # AIMessage (ChatModel)
        return raw.content.strip()
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw:    # some pipeline outputs
        first = raw[0]
        if isinstance(first, dict):
            return (first.get("generated_text") or first.get("text") or "").strip()
    return str(raw).strip()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline(vectorstore=vs, llm=llm, task="text-generation", model_id="...")
        result   = pipeline.query("What is the refund policy?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        vectorstore: FAISS,
        llm,
        task: str = "text-generation",
        model_id: str = "",
        top_k: int = 5,
        score_threshold: float = 0.25,
        rerank: bool = True,
        rerank_top_n: int = 3,
        memory_window: int = 5,
    ):
        self.vectorstore     = vectorstore
        self.llm             = llm
        self.task            = task
        self.model_id        = model_id
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self.rerank_top_n    = rerank_top_n
        self.chat_history = []
        self.reranker        = CrossEncoderReranker() if rerank else None

        logger.info(
            f"RAGPipeline ready | model={model_id} | task={task} | "
            f"top_k={top_k} | threshold={score_threshold} | rerank={rerank}"
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a question through the full RAG pipeline.

        Returns:
            answer  : str
            sources : List[Dict]  (source, page, score, snippet)
            chunks  : raw retrieved chunks
        """
        if not question.strip():
            return {"answer": "Please enter a question.", "sources": [], "chunks": []}

        # 1 ── Semantic retrieval
        docs_with_scores = similarity_search_with_threshold(
            self.vectorstore, question,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        if not docs_with_scores:
            answer = (
                "I couldn't find relevant information in the uploaded documents. "
                "Try rephrasing your question or uploading more relevant files."
            )
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            return {"answer": answer, "sources": [], "chunks": []}

        # 2 ── Optional re-ranking
        if self.reranker:
            docs_with_scores = self.reranker.rerank(
                question, docs_with_scores, top_n=self.rerank_top_n
            )
        else:
            docs_with_scores = docs_with_scores[: self.rerank_top_n]

        # 3 ── Build prompt
        prompt = _build_prompt(
            question=question,
            context=_format_context(docs_with_scores),
            chat_history=_format_history(self.chat_history),
            task=self.task,
            model_id=self.model_id,
        )

        # 4 ── Generate
        try:
            raw    = self.llm.invoke(prompt)
            answer = _extract_answer(raw)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Generation error: {e}"

        if not answer:
            answer = "The model returned an empty response. Try a different model or rephrase."

        # 5 ── Update memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        # 6 ── Build sources
        sources, seen = [], set()
        for doc, score in docs_with_scores:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                snippet = doc.page_content
                sources.append({
                    "source" : doc.metadata.get("source", "Unknown"),
                    "page"   : doc.metadata.get("page", "?"),
                    "score"  : round(score, 4),
                    "snippet": (snippet[:300] + "…") if len(snippet) > 300 else snippet,
                })

        return {"answer": answer, "sources": sources, "chunks": docs_with_scores}

    def reset_memory(self):
        self.chat_history = self.chat_history[-6:]
        logger.info("Conversation memory cleared.")

    def update_settings(self, top_k=None, score_threshold=None):
        if top_k is not None:
            self.top_k = top_k
        if score_threshold is not None:
            self.score_threshold = score_threshold
