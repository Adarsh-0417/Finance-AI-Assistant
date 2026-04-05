"""
embeddings.py
-------------
Handles:
  - Text chunking via RecursiveCharacterTextSplitter
  - Embedding generation via HuggingFace sentence-transformers (100% local)
  - FAISS vector store creation, saving, and loading
  - Semantic search with similarity threshold filtering

WHY FAISS?
----------
FAISS (Facebook AI Similarity Search) is a battle-tested C++ library with Python
bindings that performs extremely fast approximate nearest-neighbor (ANN) search
over dense float vectors. It:
  - Runs entirely on CPU (no GPU required)
  - Scales from thousands to billions of vectors
  - Supports multiple index types (Flat, IVF, HNSW) for speed/accuracy tradeoffs
  - Integrates natively with LangChain's FAISS wrapper
  - Persists to disk with a single call — perfect for multi-session use

HOW EMBEDDINGS WORK (brief):
-----------------------------
A sentence-transformer model (e.g. all-MiniLM-L6-v2) maps variable-length text
to a fixed-size dense vector (e.g. 384 dims) such that semantically similar
texts are geometrically close. We embed every chunk at index time, then embed the
user query at search time and find the nearest chunk vectors via FAISS cosine
similarity — no keyword matching required.
"""

import os
import logging
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Default constants (all overridable via arguments) ──────────────────────────
DEFAULT_CHUNK_SIZE      = 800    # characters per chunk (sweet spot for RAG)
DEFAULT_CHUNK_OVERLAP   = 150    # overlap keeps context across chunk boundaries
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_INDEX_DIR = "faiss_index"
DEFAULT_TOP_K           = 5      # number of chunks to retrieve per query
DEFAULT_SCORE_THRESHOLD = 0.25   # minimum relevance score (0–1, cosine sim)


# ── 1. Text Chunking ───────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split LangChain Documents into smaller, overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on natural boundaries in order:
        paragraph → sentence → word → character
    This preserves semantic coherence better than naive fixed-size splitting.

    Each returned chunk inherits the parent document's metadata (source, page).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Chunked {len(documents)} page(s) → {len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ── 2. Embedding Model ─────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace sentence-transformer model for generating embeddings.

    Recommended models (all free, CPU-friendly):
      - all-MiniLM-L6-v2   : 384-dim, 22 MB  — fastest, great quality  ✅ default
      - all-mpnet-base-v2  : 768-dim, 420 MB — best quality, slower
      - multi-qa-MiniLM-L6 : optimised for Q&A retrieval tasks

    Models are cached in ~/.cache/huggingface after first download.
    """
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},          # force CPU — works everywhere
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )
    logger.info("Embedding model loaded ✓")
    return embeddings


# ── 3. FAISS Vector Store ──────────────────────────────────────────────────────

def build_faiss_index(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    index_dir: str = DEFAULT_FAISS_INDEX_DIR,
) -> FAISS:
    """
    Build a FAISS vector store from document chunks and persist it to disk.

    Internally:
      1. Calls embedding_model.embed_documents() on all chunk texts
      2. Stores (embedding, chunk_text, metadata) tuples in a FAISS FlatL2 index
      3. Saves the index + docstore to `index_dir/` for reuse across sessions

    Returns the in-memory FAISS object for immediate use.
    """
    logger.info(f"Building FAISS index from {len(chunks)} chunk(s)…")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    logger.info(f"FAISS index saved to '{index_dir}/' ✓")

    return vectorstore


def load_faiss_index(
    index_dir: str = DEFAULT_FAISS_INDEX_DIR,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.

    Returns None if the index directory doesn't exist (first run).
    """
    if not os.path.exists(index_dir):
        logger.info(f"No existing FAISS index found at '{index_dir}'")
        return None

    if embedding_model is None:
        embedding_model = load_embedding_model()

    vectorstore = FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True,  # required by LangChain ≥0.1
    )
    logger.info(f"FAISS index loaded from '{index_dir}/' ✓")
    return vectorstore


# ── 4. Semantic Search with Threshold Filtering ───────────────────────────────

def similarity_search_with_threshold(
    vectorstore: FAISS,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-k most relevant chunks for a query, filtered by score.

    Uses FAISS cosine similarity (normalised embeddings → dot product = cosine).
    Scores range from 0 (unrelated) to 1 (identical).

    Chunks below `score_threshold` are discarded to prevent hallucination from
    irrelevant context being injected into the LLM prompt.

    Returns list of (Document, score) tuples, sorted by score descending.
    """
    # LangChain returns (doc, score) where score is L2 distance when unnormalised,
    # but cosine similarity when normalize_embeddings=True (our setup).
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    # Filter by threshold
    filtered = [
        (doc, score)
        for doc, score in results_with_scores
        if score >= score_threshold
    ]

    logger.info(
        f"Retrieved {len(results_with_scores)} chunks, "
        f"{len(filtered)} passed threshold {score_threshold}"
    )

    # Sort best-first
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
