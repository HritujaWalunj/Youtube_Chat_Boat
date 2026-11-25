
import os
from dotenv import load_dotenv

load_dotenv()

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "qwen3:8b"
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)

# OpenAI configuration
OPENAI_CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4o-mini"
)

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
SERVER_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# RAG configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# Application configuration
APP_NAME = "YouTube Chat Boat"
