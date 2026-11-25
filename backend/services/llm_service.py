"""Chat-model factory (Ollama or OpenAI).

Changes in this version
-----------------------
* NEW: Ollama is created with ``num_ctx`` = the chat context window and a low
  ``temperature`` (see config.py for why both matter).
* CHANGED: ``generate_response`` takes the full list of chat messages instead
  of a single string.
"""

import re

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from backend.config import (
    CONTEXT_WINDOW_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_TEMPERATURE,
    OPENAI_CHAT_MODEL,
    SERVER_OPENAI_KEY,
)


class LLMService:
    """Create the chat model for the chosen provider."""

    def __init__(self, provider: str, api_key: str = ""):
        if provider == "ollama":
            self.llm = ChatOllama(
                model=OLLAMA_CHAT_MODEL,
                base_url=OLLAMA_BASE_URL,
                # NEW: make Ollama actually allocate our context window. Without
                # this Ollama uses its own small default (4096 tokens) and
                # silently cuts the start of longer prompts, which is where our
                # instructions live.
                num_ctx=CONTEXT_WINDOW_TOKENS,
                # NEW: low randomness -> answers stay closer to the transcript.
                temperature=OLLAMA_TEMPERATURE,
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model=OPENAI_CHAT_MODEL,
                api_key=api_key or SERVER_OPENAI_KEY or None,
            )
        else:
            raise ValueError("Provider must be 'ollama' or 'openai'.")

    def generate_response(self, messages) -> str:
        """Send the chat ``messages`` (list of ``(role, text)``) and return the reply text."""
        content = self.llm.invoke(messages).content
        # Some providers return a list of content parts instead of a string.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        # Some local reasoning models (e.g. qwen3) emit <think>...</think>.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
