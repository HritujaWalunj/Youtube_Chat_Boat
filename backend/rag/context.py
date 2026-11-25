"""Context-window helpers: token estimates, history trimming, excerpt packing.

NEW in this version. Pure Python with no third-party imports, so it is easy to
read and to unit-test.

Why this module exists
----------------------
A language model can only read a limited number of *tokens* per call (its
"context window"). In a chat, three things compete for that space:

    +--------------------------------------------------------------+
    |  instructions | transcript excerpts | chat history | question |  <- prompt
    +--------------------------------------------------------------+
    |                   room reserved for the answer               |
    +--------------------------------------------------------------+
                  total must stay <= CONTEXT_WINDOW_TOKENS

If we simply kept appending messages, long chats would overflow. Instead we
keep the *most recent* messages that fit and drop the oldest ones. The full
history is still shown to the user in the browser; only what the *model*
remembers is limited.

About token counting
--------------------
Exact counts need the model's own tokenizer, which differs per model and would
add a heavy dependency. We use the common rule of thumb "1 token ~ 4
characters of English". It is an estimate, so the default budget leaves a
safety margin (see ``RESPONSE_RESERVE_TOKENS`` in config.py).
"""

from __future__ import annotations

from typing import NamedTuple

# Rule of thumb for English text.
CHARS_PER_TOKEN = 4

# Each chat message carries a few extra tokens of role markers / separators.
PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for ``text`` (rounded up)."""
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)  # ceiling division


def message_tokens(message: dict) -> int:
    """Approximate tokens used by one ``{"role", "content"}`` chat message."""
    return estimate_tokens(message["content"]) + PER_MESSAGE_OVERHEAD


def format_timestamp(seconds: int) -> str:
    """Format seconds as ``m:ss`` (or ``h:mm:ss`` for videos over an hour)."""
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class TrimResult(NamedTuple):
    """Outcome of :func:`trim_history`."""

    messages: list[dict]  # messages that fit, oldest first
    dropped: int          # how many older messages were left out
    tokens: int           # estimated tokens used by the kept messages


def trim_history(history: list[dict], budget_tokens: int) -> TrimResult:
    """Keep the newest messages whose total size fits in ``budget_tokens``.

    * Walks from the newest message backwards and stops at the first message
      that no longer fits, so the kept messages are always one unbroken,
      most-recent stretch of the conversation.
    * Never starts with an assistant reply: a reply whose question was cut off
      would confuse the model, so such leading replies are dropped as well.
    """
    kept: list[dict] = []
    used = 0
    for message in reversed(history):
        cost = message_tokens(message)
        if used + cost > budget_tokens:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    while kept and kept[0]["role"] != "user":
        used -= message_tokens(kept.pop(0))

    return TrimResult(kept, len(history) - len(kept), max(used, 0))


def pack_excerpts(
    excerpts: list[tuple[int, str]], budget_tokens: int
) -> tuple[list[tuple[int, str]], int]:
    """Fit retrieved transcript excerpts into ``budget_tokens``.

    ``excerpts`` is a list of ``(start_second, text)`` pairs ordered from most
    to least relevant. We keep them in that order until the budget runs out
    (so the *least* relevant ones are the ones dropped), then return the
    survivors sorted by video time, which reads more naturally to the model.

    If even the single best excerpt is too large, it is truncated rather than
    discarded, so the model always receives *something* to work with.

    Returns ``(kept_excerpts, tokens_used)``.
    """
    kept: list[tuple[int, str]] = []
    used = 0
    for start, text in excerpts:
        cost = estimate_tokens(text) + PER_MESSAGE_OVERHEAD  # +label like "[3:07]"
        if used + cost > budget_tokens:
            if not kept and budget_tokens > PER_MESSAGE_OVERHEAD:
                max_chars = (budget_tokens - PER_MESSAGE_OVERHEAD) * CHARS_PER_TOKEN
                text = text[:max_chars]
                cost = estimate_tokens(text) + PER_MESSAGE_OVERHEAD
                kept.append((start, text))
                used += cost
            break
        kept.append((start, text))
        used += cost
    kept.sort(key=lambda item: item[0])
    return kept, used


def build_retrieval_query(question: str, history: list[dict], short_words: int = 6) -> str:
    """Choose the text used to search the transcript.

    Follow-up questions such as "why?" or "and the second one?" contain almost
    no searchable meaning on their own. When the new question is that short
    and a previous question exists, we search with both, so retrieval stays on
    topic. Longer questions are searched as-is.

    (A more powerful but slower alternative is to ask the LLM to rewrite the
    question into a stand-alone one; that costs an extra model call per turn.)
    """
    if history and len(question.split()) <= short_words:
        for message in reversed(history):
            if message["role"] == "user":
                return f"{message['content']} {question}"
    return question
