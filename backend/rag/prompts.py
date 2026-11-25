"""Prompt used by the RAG pipeline.

Changes in this version
-----------------------
* CHANGED: the old single ``ChatPromptTemplate`` string is replaced by a
  proper *system* message + earlier chat turns + the new *human* message.
  Chat models follow rules given in the system role more reliably, which
  matters for small local models that tend to drift into outside knowledge.
* NEW: the model also receives the previous messages of the chat, so
  follow-up questions ("what did he say after that?") work.
* NEW: every transcript excerpt starts with its timestamp (``[3:07]``) and the
  model may cite it.
* The messages are plain ``(role, text)`` tuples, which LangChain chat models
  accept directly, so this file needs no LangChain import.
"""

# The exact sentence the model must use when the transcript has no answer.
# chain.py looks for it to decide whether to hide the timestamp links.
NOT_FOUND_MESSAGE = "I couldn't find this information in the video transcript."

# ``{context}`` is replaced by the retrieved transcript excerpts (see
# ``build_messages``). Written as plain text with ``str.replace`` rather than
# ``str.format`` so that curly braces in a transcript can never break it.
SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about ONE YouTube video, using ONLY its transcript.

Rules:
1. Base every statement on the transcript excerpts below. Each excerpt begins with its video timestamp, like [3:07].
2. If the excerpts do not contain the answer, reply with exactly: "{NOT_FOUND_MESSAGE}" Do not guess, and do not use outside knowledge.
3. The transcript is automatic captions, so names and technical terms may be misspelled. You may point out an obvious misspelling, but never invent content.
4. Earlier messages in the chat are only there so you can understand follow-up questions (for example what "it" refers to). Facts must still come from the excerpts.
5. Be concise. Use short paragraphs or bullet points, and cite timestamps like [3:07] when it helps.

Transcript excerpts:
{{context}}"""


def prompt_overhead_text() -> str:
    """Instruction text with an empty context; used to estimate its token cost."""
    return SYSTEM_PROMPT.replace("{context}", "")


def build_messages(context: str, history: list[dict], question: str) -> list[tuple[str, str]]:
    """Assemble the full prompt as ``[(role, text), ...]``.

    Order matters: instructions + transcript first, then earlier turns, then
    the new question last (models pay the most attention to the end).
    """
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT.replace("{context}", context))]
    for turn in history:
        role = "human" if turn["role"] == "user" else "ai"
        messages.append((role, turn["content"]))
    messages.append(("human", question))
    return messages
