"""FastAPI entry point: JSON API under /api, frontend served at /."""

import os
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import LLM_PROVIDER, SERVER_OPENAI_KEY, MAX_SESSIONS
from backend.services.transcript_service import (
    TranscriptService, InvalidYouTubeURL, TranscriptUnavailable,
)
from backend.services.embedding_service import EmbeddingService
from backend.services.llm_service import LLMService
from backend.services.vector_service import VectorService
from backend.rag.chain import RAGChain

app = FastAPI(title="YouTube Chat Boat", version="1.0.0")

transcript_service = TranscriptService()
vector_service = VectorService()
rag_chain = RAGChain(vector_service)

# session_id -> {"store", "provider", "video_id"}; oldest evicted first.
sessions: "OrderedDict[str, dict]" = OrderedDict()


class LoadRequest(BaseModel):
    url: str
    provider: str = "ollama"
    api_key: str = ""


class ChatRequest(BaseModel):
    session_id: str
    question: str
    api_key: str = ""


def _check_provider(provider: str, api_key: str):
    if provider not in ("ollama", "openai"):
        raise HTTPException(400, "Provider must be 'ollama' or 'openai'.")
    if provider == "openai" and not (api_key or SERVER_OPENAI_KEY):
        raise HTTPException(400, "Please enter your OpenAI API key.")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"default_provider": LLM_PROVIDER, "server_openai_key": bool(SERVER_OPENAI_KEY)}


@app.post("/api/load")
def load_video(req: LoadRequest):
    _check_provider(req.provider, req.api_key)
    try:
        data = transcript_service.fetch_transcript(req.url)
        embeddings = EmbeddingService(req.provider, req.api_key).embeddings
        store = vector_service.create_vector_store(data["segments"], embeddings)
    except (InvalidYouTubeURL, TranscriptUnavailable) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Could not index the video: {exc}")

    session_id = uuid.uuid4().hex
    sessions[session_id] = {"store": store, "provider": req.provider, "video_id": data["video_id"]}
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)

    return {
        "session_id": session_id,
        "video_id": data["video_id"],
        "chunks": store.index.ntotal,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session expired. Please load the video again.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    _check_provider(session["provider"], req.api_key)
    try:
        embeddings = EmbeddingService(session["provider"], req.api_key).embeddings
        llm = LLMService(session["provider"], req.api_key)
        return rag_chain.process_question(session["store"], embeddings, llm, req.question)
    except Exception as exc:
        raise HTTPException(500, f"Could not answer: {exc}")


# Serve the frontend last so /api/* routes take priority.
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
