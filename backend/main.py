"""FastAPI entry point: JSON API under /api, frontend served at /.

Endpoints
---------
GET    /api/health              liveness probe (used by the Docker healthcheck)
GET    /api/config              settings the UI needs (defaults, limits, model names)
POST   /api/load                download captions, build the search index, open a session
POST   /api/chat                answer a question about a loaded video
DELETE /api/session/{id}        free a session's memory (NEW)

Changes in this version
-----------------------
* NEW: ``/api/chat`` accepts the chat ``history`` (see ``rag/context.py`` for
  how it is trimmed to the context window) and returns a ``usage`` summary.
* NEW: ``/api/load`` also returns the video ``title``.
* NEW: ``/api/config`` also returns ``max_chats``, ``context_window`` and the
  model names; ``DELETE /api/session/{id}`` frees an index when the UI drops a chat.
* CHANGED: a session that is used moves to the "newest" end, so sessions in
  active use are not evicted before idle ones.

Where the chat history lives
----------------------------
The *browser* stores each chat (its messages and video link) and sends the
recent messages with every question. The server keeps only the search index
per video. That keeps the server simple and lets a chat survive a server
restart: if the index is gone the UI silently re-indexes the video.
"""

import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import (
    CONTEXT_WINDOW_TOKENS,
    LLM_PROVIDER,
    MAX_CHATS,
    MAX_QUESTION_CHARS,
    MAX_SESSIONS,
    OLLAMA_CHAT_MODEL,
    OPENAI_CHAT_MODEL,
    SERVER_OPENAI_KEY,
)
from backend.services.transcript_service import (
    TranscriptService, InvalidYouTubeURL, TranscriptUnavailable,
)
from backend.services.embedding_service import EmbeddingService
from backend.services.llm_service import LLMService
from backend.services.vector_service import VectorService
from backend.rag.chain import RAGChain

app = FastAPI(title="YouTube Chat Boat", version="1.1.0")

# Shared, stateless helpers (created once at start-up).
transcript_service = TranscriptService()
vector_service = VectorService()
rag_chain = RAGChain(vector_service)

# session_id -> {"store", "provider", "video_id"}; least recently used evicted first.
sessions: "OrderedDict[str, dict]" = OrderedDict()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class LoadRequest(BaseModel):
    """Body of POST /api/load."""
    url: str
    provider: str = "ollama"
    api_key: str = ""


class ChatMessage(BaseModel):
    """One earlier message of the conversation (NEW)."""
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    """Body of POST /api/chat."""
    session_id: str
    question: str
    api_key: str = ""
    # NEW: earlier messages, oldest first, NOT including ``question``.
    history: list[ChatMessage] = Field(default_factory=list, max_length=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_provider(provider: str, api_key: str):
    """Reject unknown providers and OpenAI requests that have no API key."""
    if provider not in ("ollama", "openai"):
        raise HTTPException(400, "Provider must be 'ollama' or 'openai'.")
    if provider == "openai" and not (api_key or SERVER_OPENAI_KEY):
        raise HTTPException(400, "Please enter your OpenAI API key.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/api/config")
def config():
    """Settings the frontend needs at start-up."""
    return {
        "default_provider": LLM_PROVIDER,
        "server_openai_key": bool(SERVER_OPENAI_KEY),
        # NEW:
        "max_chats": MAX_CHATS,
        "context_window": CONTEXT_WINDOW_TOKENS,
        "ollama_model": OLLAMA_CHAT_MODEL,
        "openai_model": OPENAI_CHAT_MODEL,
    }


@app.post("/api/load")
def load_video(req: LoadRequest):
    """Fetch captions, build the search index and return a new session id."""
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
        sessions.popitem(last=False)  # drop the least recently used session

    return {
        "session_id": session_id,
        "video_id": data["video_id"],
        "title": transcript_service.fetch_title(data["video_id"]),  # NEW (may be None)
        "chunks": store.index.ntotal,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Answer one question using the video's transcript and the chat so far."""
    session = sessions.get(req.session_id)
    if not session:
        # The UI reacts to this 404 by re-indexing the video and retrying once.
        raise HTTPException(404, "Session expired. Please load the video again.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if len(req.question) > MAX_QUESTION_CHARS:  # NEW
        raise HTTPException(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    _check_provider(session["provider"], req.api_key)

    sessions.move_to_end(req.session_id)  # mark as recently used
    try:
        embeddings = EmbeddingService(session["provider"], req.api_key).embeddings
        llm = LLMService(session["provider"], req.api_key)
        return rag_chain.process_question(
            session["store"],
            embeddings,
            llm,
            req.question.strip(),
            history=[m.model_dump() for m in req.history],  # NEW
        )
    except Exception as exc:
        raise HTTPException(500, f"Could not answer: {exc}")


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Free the search index of a chat the user deleted (NEW). Always succeeds."""
    return {"deleted": sessions.pop(session_id, None) is not None}


# Serve the frontend last so /api/* routes take priority.
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
