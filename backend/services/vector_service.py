"""Timestamp-aware chunking, FAISS indexing and retrieval.

(No logic changes in this version; docstrings and comments were added.)

Pipeline: caption segments -> ``build_documents`` (chunks that remember the
second they start at) -> FAISS index (vector search) -> ``retrieve``.
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


def build_documents(segments: list[dict]) -> list[Document]:
    """Group caption segments into chunks; each chunk remembers its start second.

    ``segments`` look like ``{"text": "...", "start": 12.3, "duration": 2.1}``.
    A chunk is closed once adding the next segment would exceed ``CHUNK_SIZE``
    characters; the last ``CHUNK_OVERLAP`` characters of segments are carried
    into the next chunk so a sentence cut at a boundary is still found.
    """
    docs: list[Document] = []
    cur: list[dict] = []   # segments of the chunk being built
    size = 0               # characters in ``cur``

    def flush():
        """Turn the segments collected so far into one Document."""
        text = " ".join(s["text"].replace("\n", " ") for s in cur)
        docs.append(Document(page_content=text, metadata={"start": int(cur[0]["start"])}))

    for seg in segments:
        n = len(seg["text"]) + 1  # +1 for the joining space
        if cur and size + n > CHUNK_SIZE:
            flush()
            keep, total = [], 0
            for s in reversed(cur):  # carry over a small overlap
                total += len(s["text"]) + 1
                if total > CHUNK_OVERLAP:
                    break
                keep.insert(0, s)
            cur = keep
            size = sum(len(s["text"]) + 1 for s in cur)
        cur.append(seg)
        size += n
    if cur:
        flush()
    return docs


class VectorService:
    """Builds and queries the per-video FAISS index."""

    def create_vector_store(self, segments: list[dict], embeddings) -> FAISS:
        """Chunk the transcript and embed every chunk into a new FAISS index."""
        docs = build_documents(segments)
        if not docs:
            raise ValueError("Transcript is empty or could not be split into chunks.")
        return FAISS.from_documents(docs, embeddings)

    def retrieve(self, store: FAISS, embeddings, question: str) -> list[Document]:
        """Return the ``RETRIEVER_K`` chunks most similar to ``question`` (best first)."""
        vector = embeddings.embed_query(question)
        return store.similarity_search_by_vector(vector, k=RETRIEVER_K)
