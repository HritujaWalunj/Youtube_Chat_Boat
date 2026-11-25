import re

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from backend.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OPENAI_CHAT_MODEL,
    SERVER_OPENAI_KEY,
)


class LLMService:
    """Create the chat model for the chosen provider."""

    def __init__(self, provider: str, api_key: str = ""):
        if provider == "ollama":
            self.llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model=OPENAI_CHAT_MODEL,
                api_key=api_key or SERVER_OPENAI_KEY or None,
            )
        else:
            raise ValueError("Provider must be 'ollama' or 'openai'.")

    def generate_response(self, prompt: str) -> str:
        content = self.llm.invoke(prompt).content
        # Some local reasoning models (e.g. qwen3) emit <think>...</think>.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
