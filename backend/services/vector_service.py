"""Timestamp-aware chunking, FAISS indexing and retrieval."""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


def build_documents(segments: list[dict]) -> list[Document]:
    """Group caption segments into chunks; each chunk remembers its start second."""
    docs: list[Document] = []
    cur: list[dict] = []
    size = 0

    def flush():
        text = " ".join(s["text"].replace("\n", " ") for s in cur)
        docs.append(Document(page_content=text, metadata={"start": int(cur[0]["start"])}))

    for seg in segments:
        n = len(seg["text"]) + 1
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
    def create_vector_store(self, segments: list[dict], embeddings) -> FAISS:
        docs = build_documents(segments)
        if not docs:
            raise ValueError("Transcript is empty or could not be split into chunks.")
        return FAISS.from_documents(docs, embeddings)

    def retrieve(self, store: FAISS, embeddings, question: str) -> list[Document]:
        vector = embeddings.embed_query(question)
        return store.similarity_search_by_vector(vector, k=RETRIEVER_K)
