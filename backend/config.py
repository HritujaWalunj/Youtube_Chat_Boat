"""Central configuration for YouTube Chat Boat.

Every setting is read from an environment variable (or from the ``.env`` file in
the project root) and falls back to a sensible default, so the app runs with
no configuration at all. See ``.env.example`` for a documented template.

Changes in this version
-----------------------
* NEW: ``MAX_CHATS`` ................ how many chats a user keeps (default 3).
* NEW: ``CONTEXT_WINDOW_TOKENS`` .... per-chat context budget (default 10,000).
* NEW: ``RESPONSE_RESERVE_TOKENS`` .. room kept free for the model's answer.
* NEW: ``OLLAMA_TEMPERATURE`` ....... lower randomness = fewer made-up facts.
* NEW: ``MAX_QUESTION_CHARS`` ....... guard against oversized requests.
All other settings are unchanged.
"""

import os

from dotenv import load_dotenv

# Read variables from a local ``.env`` file (if present) into ``os.environ``.
load_dotenv()

# ---------------------------------------------------------------------------
# LLM provider (pre-existing)
# ---------------------------------------------------------------------------
# Provider pre-selected in the UI: "ollama" (local, free) or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ---------------------------------------------------------------------------
# Ollama (pre-existing, plus NEW temperature)
# ---------------------------------------------------------------------------
# Where the Ollama server listens. Inside Docker this is overridden to the
# ``ollama`` service by docker-compose.yml.
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

# Model that writes the answers. Any text model from https://ollama.com/library
# works; see the README section "Choosing an Ollama model".
OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "qwen3:8b"
)

# Model that turns transcript chunks / questions into vectors for search.
OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)

# NEW: sampling temperature for Ollama. Ollama's own default is higher
# (~0.8) which encourages creative, *less factual* answers. For question
# answering over a transcript we want the model to stick to the text, so we
# default to a low value. (Avoid exactly 0 for Qwen3 "thinking" models; the
# Qwen team warns that fully greedy decoding can cause repetition loops.)
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# OpenAI (pre-existing)
# ---------------------------------------------------------------------------
OPENAI_CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4o-mini"
)

# Optional server-wide key. If empty, each user must type their own key in the
# UI (it is then sent with each request and never stored by the server).
SERVER_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# ---------------------------------------------------------------------------
# Server-side sessions (pre-existing)
# ---------------------------------------------------------------------------
# Each loaded video keeps its search index in server memory as a "session".
# When more than MAX_SESSIONS exist, the oldest one is dropped.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))

# ---------------------------------------------------------------------------
# Chat history & context window (NEW)
# ---------------------------------------------------------------------------
# How many chats each user keeps in the sidebar. Starting a new chat beyond
# this limit removes the oldest one. (Enforced in the browser, which owns the
# history; the server is told to free the removed chat's search index.)
MAX_CHATS = int(os.getenv("MAX_CHATS", "3"))

# The "context window" of a chat: the maximum number of tokens we put into a
# single model call (instructions + transcript excerpts + earlier messages +
# the new question + room for the reply). 10,000 is a good balance:
#   * big enough for ~15-20 back-and-forth turns plus several transcript
#     excerpts, and
#   * small enough to stay fast and cheap with a local model (the memory the
#     model needs grows with this number).
# This same value is passed to Ollama as ``num_ctx`` so Ollama really
# allocates that much (otherwise it silently uses its own, smaller default
# and cuts off the *start* of the prompt, which includes our instructions).
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "10000"))

# Tokens kept free inside the window for the model's reply (and, for
# "thinking" models such as qwen3 / deepseek-r1, for their reasoning text).
RESPONSE_RESERVE_TOKENS = int(os.getenv("RESPONSE_RESERVE_TOKENS", "1500"))

# Longest question (in characters) the API accepts.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# ---------------------------------------------------------------------------
# RAG (retrieval) settings (pre-existing)
# ---------------------------------------------------------------------------
# Transcript is cut into chunks of about CHUNK_SIZE characters, neighbouring
# chunks sharing CHUNK_OVERLAP characters, and the RETRIEVER_K chunks most
# similar to the question are handed to the model.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# ---------------------------------------------------------------------------
# Application (pre-existing)
# ---------------------------------------------------------------------------
APP_NAME = "YouTube Chat Boat"
