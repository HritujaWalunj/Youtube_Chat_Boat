
"""Prompt templates used by the RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate


# Instructions for answering questions using YouTube transcript context.
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions about YouTube videos.

    Use ONLY the provided transcript context to answer the question.

    If the answer is not available in the transcript, clearly say:
    "I couldn't find this information in the video transcript."

    Do not make up facts or use outside knowledge.

    Transcript Context:
    {context}

    User Question:
    {question}

    Answer:
    """
)
