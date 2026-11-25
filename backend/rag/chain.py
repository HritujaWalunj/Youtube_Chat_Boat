"""RAG chain: retrieve transcript excerpts, build a budgeted prompt, ask the LLM.

Changes in this version
-----------------------
* NEW: accepts the chat ``history`` so follow-up questions work.
* NEW: enforces the per-chat context window (see ``rag/context.py``): the
  transcript excerpts and the chat history are trimmed to fit.
* NEW: returns a ``usage`` summary that the UI shows as a small meter.
* CHANGED: timestamp links are only returned when the model actually found an
  answer (previously they were shown even for "I couldn't find this...").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config import CONTEXT_WINDOW_TOKENS, RESPONSE_RESERVE_TOKENS
from backend.rag.context import (
    PER_MESSAGE_OVERHEAD,
    build_retrieval_query,
    estimate_tokens,
    format_timestamp,
    pack_excerpts,
    trim_history,
)
from backend.rag.prompts import NOT_FOUND_MESSAGE, build_messages, prompt_overhead_text

if TYPE_CHECKING:  # only for type hints; avoids importing heavy libraries here
    from backend.services.llm_service import LLMService
    from backend.services.vector_service import VectorService

# Share of the available space transcript excerpts may use at most. Whatever
# they do not use is given to chat history (usually the excerpts are small).
EXCERPT_SHARE = 0.6


def is_not_found(answer: str) -> bool:
    """True if the model replied with the standard "not in the transcript" text."""
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower()


class RAGChain:
    """Connects retrieval (FAISS), the prompt and the LLM."""

    def __init__(self, vector_service: "VectorService"):
        self.vector_service = vector_service

    def process_question(
        self,
        store,
        embeddings,
        llm: "LLMService",
        question: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Answer ``question`` about the video held in ``store``.

        ``history`` is the earlier conversation as ``[{"role": "user" |
        "assistant", "content": "..."}]``, oldest first, *not* including the
        new question.

        Returns ``{"answer", "sources", "usage"}`` where ``sources`` are the
        start seconds of the excerpts the answer was based on and ``usage``
        describes how full the context window was.
        """
        history = history or []

        # 1. Retrieve the transcript chunks most similar to the question.
        #    (Short follow-ups are searched together with the previous question.)
        query = build_retrieval_query(question, history)
        documents = self.vector_service.retrieve(store, embeddings, query)

        # 2. Work out how much room we have inside the context window.
        available = max(CONTEXT_WINDOW_TOKENS - RESPONSE_RESERVE_TOKENS, 0)
        fixed = (
            estimate_tokens(prompt_overhead_text())      # rules text
            + estimate_tokens(question)
            + 2 * PER_MESSAGE_OVERHEAD                   # system + new human message
        )
        remaining = max(available - fixed, 0)

        # 3. Fit excerpts first (they ground the answer), then give all leftover
        #    room to the most recent chat history.
        excerpts = [(d.metadata["start"], d.page_content) for d in documents]
        kept_excerpts, excerpt_tokens = pack_excerpts(excerpts, int(remaining * EXCERPT_SHARE))
        trimmed = trim_history(history, remaining - excerpt_tokens)

        # 4. Build the prompt and ask the model.
        context = "\n\n".join(
            f"[{format_timestamp(start)}] {text}" for start, text in kept_excerpts
        )
        messages = build_messages(context, trimmed.messages, question)
        answer = llm.generate_response(messages)

        # 5. Only link timestamps if the model found something.
        sources = [] if is_not_found(answer) else sorted({start for start, _ in kept_excerpts})

        return {
            "answer": answer,
            "sources": sources,
            "usage": {
                "used": fixed + excerpt_tokens + trimmed.tokens,  # estimated prompt tokens
                "limit": CONTEXT_WINDOW_TOKENS,
                "dropped": trimmed.dropped,  # older messages the model no longer sees
            },
        }
