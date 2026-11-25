"""Prompt used by the RAG pipeline.

Changes in this version
-----------------------
* CHANGED: the old single ``ChatPromptTemplate`` string is replaced by a
  proper *system* message + earlier chat turns + the new *human* message.
  Chat models follow rules given in the system role more reliably, which
  matters for small local models that tend to drift into outside knowledge.
* NEW: the model also receives the previous messages of the chat, so
  follow-up questions ("what did he say after that?") work.
* CHANGED: transcript excerpts are sent WITHOUT timestamps and the model is told
  never to write any. The timestamp links are shown once, at the end of the
  message, by the UI (see ``sources`` in chain.py).
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
1. Base every statement on the transcript excerpts below.
2. If the excerpts do not contain the answer, reply with exactly: "{NOT_FOUND_MESSAGE}" Do not guess, and do not use outside knowledge.
3. The transcript is automatic captions, so names and technical terms may be misspelled. You may point out an obvious misspelling, but never invent content.
4. Earlier messages in the chat are only there so you can understand follow-up questions (for example what "it" refers to). Facts must still come from the excerpts.
5. Format every answer so it is easy to scan, using plain Markdown only (the exact reply from rule 2 stays as plain text):
   - Begin with short paragraph that directly answers the question.
   - Leave a blank line, then give 3 to 6 bullet points. One idea per bullet, each on its own line starting with "- ".
   - Put the important words in **bold**: key terms, names, numbers and the main idea of each bullet.
   - Use a numbered list ("1.", "2.") only for steps or ordered sequences.
   - Be concise. Do not repeat the question.
   Example of the layout:

   The video explains how **transformers** understand the meaning of words.

   - **Embeddings:** every word becomes a vector (a list of numbers) that captures its meaning.
   - **Attention:** lets each word take in meaning from the words around it.
   - **Context size:** the model can only read a limited number of tokens at a time.
6. Never write timestamps, times or links in your answer (no [3:07], no 3:07, no URLs). They are added separately.

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