"""RAG chain: retrieve transcript excerpts, build a budgeted prompt, ask the LLM.

Changes in this version
-----------------------
* NEW: accepts the chat ``history`` so follow-up questions work.
* NEW: enforces the per-chat context window (see ``rag/context.py``): the
  transcript excerpts and the chat history are trimmed to fit.
* NEW: returns a ``usage`` summary that the UI shows as a small meter.
* CHANGED: timestamp links are only returned when the model actually found an
  answer (previously they were shown even for "I couldn't find this...").
* CHANGED: no timestamps inside the answer text. They are not sent to the model
  and any that still slip through are removed by ``strip_timestamps``. The
  timestamps are returned separately in ``sources`` and the UI shows them once,
  at the end of the message.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.config import CONTEXT_WINDOW_TOKENS, RESPONSE_RESERVE_TOKENS
from backend.rag.context import (
    PER_MESSAGE_OVERHEAD,
    build_retrieval_query,
    estimate_tokens,
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


# One timestamp such as 3:42 or 1:02:05.
_TS = r"(?:\d+:)?\d{1,2}:\d{2}"
# [3:42]  [[3:42]]  [3:42](https://...)  [[3:42]](https://...)  [3:42, 6:30]  [3:42-4:10]
_TS_BRACKET = re.compile(
    r"\[{1,2}\s*" + _TS + r"(?:\s*(?:,|;|-|\u2013|to)\s*" + _TS + r")*\s*\]{1,2}(?:\([^)]*\))?"
)


def strip_timestamps(answer: str) -> str:
    """Remove bracketed timestamps (and their links) from the answer text.

    The prompt already tells the model not to write them; this is the safety
    net for small models that do it anyway. Only the bracket formats above are
    removed, so ordinary text such as "a 3:1 ratio" is left alone.
    """
    text = _TS_BRACKET.sub("", answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)      # "word ." -> "word."
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)         # double spaces (keeps list indent)
    text = re.sub(r"(?m)^\s*[-*\u2022]\s*$\n?", "", text)  # bullets left empty
    return re.sub(r"[ \t]+$", "", text, flags=re.M).strip()


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
        # Timestamps are deliberately NOT put in the prompt (see prompts.py).
        context = "\n\n".join(text for _, text in kept_excerpts)
        messages = build_messages(context, trimmed.messages, question)
        answer = llm.generate_response(messages)

        # 5. Only link timestamps if the model found something. They are
        #    returned here and shown at the end of the message, never inline.
        sources = [] if is_not_found(answer) else sorted({start for start, _ in kept_excerpts})
        answer = strip_timestamps(answer)

        return {
            "answer": answer,
            "sources": sources,
            "usage": {
                "used": fixed + excerpt_tokens + trimmed.tokens,  # estimated prompt tokens
                "limit": CONTEXT_WINDOW_TOKENS,
                "dropped": trimmed.dropped,  # older messages the model no longer sees
            },
        }