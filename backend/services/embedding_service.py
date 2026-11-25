"""Embedding-model factory (Ollama or OpenAI).

Changes in this version
-----------------------
* NEW: ``NomicPrefixEmbeddings`` wrapper. The ``nomic-embed-text`` model was
  trained with task prefixes ("search_document: ..." for stored text and
  "search_query: ..." for questions). Without them it still works but finds
  the right transcript passages less reliably, and wrong passages are a main
  cause of wrong answers. LangChain/Ollama do not add the prefixes for you.
"""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from backend.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    SERVER_OPENAI_KEY,
)


class NomicPrefixEmbeddings(Embeddings):
    """Wrap another embeddings object and add nomic-embed-text's task prefixes.

    Uses composition (not subclassing) so the prefix can never be applied
    twice, whatever the wrapped class does internally.
    """

    DOC_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(self, inner: Embeddings):
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed transcript chunks that will be stored in the index."""
        return self.inner.embed_documents([self.DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question used to search the index."""
        return self.inner.embed_query(self.QUERY_PREFIX + text)


class EmbeddingService:
    """Create the embedding model for the chosen provider."""

    def __init__(self, provider: str, api_key: str = ""):
        if provider == "ollama":
            embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
            # NEW: only nomic models need (and understand) these prefixes.
            if OLLAMA_EMBEDDING_MODEL.lower().startswith("nomic-embed-text"):
                embeddings = NomicPrefixEmbeddings(embeddings)
            self.embeddings = embeddings
        elif provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=api_key or SERVER_OPENAI_KEY or None,
            )
        else:
            raise ValueError("Provider must be 'ollama' or 'openai'.")
