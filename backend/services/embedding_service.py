from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from backend.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    SERVER_OPENAI_KEY,
)


class EmbeddingService:
    """Create the embedding model for the chosen provider."""

    def __init__(self, provider: str, api_key: str = ""):
        if provider == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
        elif provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=api_key or SERVER_OPENAI_KEY or None,
            )
        else:
            raise ValueError("Provider must be 'ollama' or 'openai'.")
