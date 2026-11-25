"""RAG chain connecting retrieval, prompt and LLM."""

from backend.services.vector_service import VectorService
from backend.services.llm_service import LLMService
from backend.rag.prompts import RAG_PROMPT


class RAGChain:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service

    def process_question(self, store, embeddings, llm: LLMService, question: str) -> dict:
        documents = self.vector_service.retrieve(store, embeddings, question)
        context = "\n\n".join(d.page_content for d in documents)
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = llm.generate_response(prompt)
        sources = sorted({d.metadata["start"] for d in documents})
        return {"answer": answer, "sources": sources}
