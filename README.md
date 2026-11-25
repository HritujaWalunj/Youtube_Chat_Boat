# YouTube Chat Boat

Chat with any YouTube video. Paste a link, and ask questions; answers are
generated **only from the video's English captions** and link back to the
moment in the video they came from.

It is a small RAG (retrieval-augmented generation) app: FastAPI backend,
FAISS vector search, and a plain HTML/CSS/JS frontend. It runs on a local
[Ollama](https://ollama.com) model (free, private) or on OpenAI.

## Features

- Ask questions about a video and get answers with clickable timestamps
- **Chat history:** the 3 most recent chats are kept in a sidebar (one video per chat)
- **Follow-up questions** work, because each chat remembers its conversation
- **10,000-token context window per chat**, with a meter showing how full it is
- Clean ChatGPT-style interface: light/dark mode, works on phones
- Choose Ollama (local) or OpenAI per chat

## How it works

```
 YouTube link ──► captions (with timestamps) ──► chunks (~1000 chars) ──► embeddings ──► FAISS index
                                                                                              │
 your question (+ recent chat) ──► embed question ──► find the 4 most similar chunks ◄────────┘
                                                              │
                          instructions + excerpts [m:ss] + chat history + question  (≤ 10k tokens)
                                                              │
                                                       LLM (Ollama / OpenAI) ──► answer + timestamps
```

## Project structure

```
Youtube_Chat_Boat/
├── backend/
│   ├── main.py                    API routes (FastAPI) + serves the frontend
│   ├── config.py                  all settings, read from environment / .env
│   ├── rag/
│   │   ├── chain.py               retrieve → fit to context window → ask the LLM
│   │   ├── context.py             token estimates, history trimming (new)
│   │   └── prompts.py             system prompt + prompt assembly
│   └── services/
│       ├── transcript_service.py  YouTube captions and video title
│       ├── vector_service.py      chunking, FAISS index, retrieval
│       ├── embedding_service.py   Ollama / OpenAI embeddings
│       ├── llm_service.py         Ollama / OpenAI chat model
│       └── check_transcript.py    manual test script for captions
├── frontend/
│   ├── index.html  style.css  script.js   the chat interface
│   └── markdown.js                        safe rendering of **bold**, lists, code
├── tests/test_context_window.py   unit tests for history/context logic
├── Dockerfile  docker-compose.yml  requirements.txt
└── .env.example                   documented settings template
```

## Quick start

### Option A: Docker (easiest)

Needs Docker with Compose. Ollama and the models are set up for you.

```bash
cp .env.example .env
docker compose up --build
```

The first start downloads the models (`qwen3:8b` is about 5 GB), so wait for the
`ollama-pull` container to finish. Then open **http://localhost:8000**.


### Option B: run locally

Needs Python 3.12 and [Ollama](https://ollama.com/download).

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional
uvicorn backend.main:app --reload
```

Open **http://localhost:8000**.

To use OpenAI instead, pick "OpenAI" on the start screen and paste your key
(kept only in that browser tab), or set `OPENAI_API_KEY` in `.env`.

## Using the app

1. Paste a YouTube link (the video needs **English captions**) and press **Start chat**.
2. Ask questions. Press **Enter** to send, **Shift+Enter** for a new line.
3. Click a timestamp chip or a `[3:07]` link to open the video at that moment.
   **Watch video** (top right) shows the player inside the page.
4. **New chat** starts another video. You keep at most **3 chats**; starting a
   fourth removes the least recently used one (the screen warns you first).
   The × next to a chat deletes it.

## Chat history and the 10k context window

**History.** Chats are saved in your browser (`localStorage`), so they survive
reloads and server restarts, and are private to that browser. If the server has
forgotten a video (after a restart), the app silently re-indexes it from the saved
link and carries on. Clearing your browser data deletes the history.

**Context window.** A model can only read so many tokens per question. Each
question uses a budget of `CONTEXT_WINDOW_TOKENS` (default **10,000**):

| Part | Share of the budget |
|---|---|
| Room kept free for the answer (`RESPONSE_RESERVE_TOKENS`) | 1,500 |
| Instructions + your new question | about 400 |
| Transcript excerpts | up to 60% of what remains (normally about 1,000) |
| Earlier chat messages | everything left over, newest first |

When a long chat exceeds the budget, the **oldest messages stop being sent to the
model** (they stay visible on screen). The meter in the top bar turns amber and a
note appears. Tokens are estimated at about 4 characters each, which is why a
margin is kept. For Ollama, the same number is sent as `num_ctx`; without this,
Ollama defaults to a smaller window and silently cuts the beginning of long prompts.

Why 10k: it fits roughly 15–20 question/answer turns plus excerpts, but stays
fast on a laptop (the memory a local model needs grows with context size).
Raise or lower it with `CONTEXT_WINDOW_TOKENS`.

## Configuration

Set in `.env` (see `.env.example`). All are optional.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Provider pre-selected in the UI (`ollama` / `openai`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | Model that writes answers |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model used for search |
| `OLLAMA_TEMPERATURE` | `0.2` | Lower = fewer invented details |
| `OPENAI_API_KEY` | empty | Server-wide key (otherwise each user enters theirs) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI answer model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI search model |
| `MAX_CHATS` | `3` | Chats kept per browser |
| `CONTEXT_WINDOW_TOKENS` | `10000` | Tokens per model call |
| `RESPONSE_RESERVE_TOKENS` | `1500` | Space kept for the reply |
| `MAX_QUESTION_CHARS` | `2000` | Longest accepted question |
| `MAX_SESSIONS` | `50` | Videos kept indexed in server memory |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Transcript chunking (characters) |
| `RETRIEVER_K` | `4` | Excerpts retrieved per question |

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/config` | Defaults, limits and model names for the UI |
| `POST /api/load` | `{url, provider, api_key}` → `{session_id, video_id, title, chunks}` |
| `POST /api/chat` | `{session_id, question, history, api_key}` → `{answer, sources, usage}` |
| `DELETE /api/session/{id}` | Free a video's index |

`history` is `[{role: "user" \| "assistant", content}]`, oldest first, excluding the new
question. `usage` is `{used, limit, dropped}` (estimated prompt tokens, window size,
messages left out). Interactive docs: http://localhost:8000/docs

## Choosing an Ollama model 

Wrong or invented answers with a local model usually come from a mix of:

1. **Model size.** 7–8B models quantized to 4 bits follow "use only the context"
   less strictly than a large hosted model such as `gpt-4o-mini`.
2. **Randomness.** Ollama's default temperature is high. This version defaults to `0.2`.
3. **Truncated prompts.** Ollama's default context is small and it silently cuts the
   *start* of a long prompt (where the instructions are). This version sets `num_ctx`.
4. **Weak retrieval.** If the wrong passages are retrieved, even a good model answers
   wrongly. `nomic-embed-text` expects `search_document:` / `search_query:` prefixes,
   which this version now adds.

You are not limited to Qwen. Any text model from the
[Ollama library](https://ollama.com/library) works: set `OLLAMA_CHAT_MODEL` and
run `ollama pull <name>`. Worth trying for question answering (sizes are Q4 downloads; check the
library for current tags):

| Model | Approx. size | Notes |
|---|---|---|
| `llama3.1:8b` | ~5 GB | Same size class as qwen3:8b, widely used for RAG, 128K context |
| `granite4:3b` / `granite4.2:8b` | 2–5 GB | IBM; aimed at enterprise RAG and tool use |
| `gemma3:12b` (or newer `gemma4`) | ~8 GB | Strong for its size; good if you have 12 GB VRAM |
| `mistral-small3.2:24b` | ~15 GB | Needs a 16–24 GB GPU or a 32 GB Mac |
| `gpt-oss:20b` | ~14 GB | OpenAI open weights; runs on 16 GB RAM |
| `llama3.3:70b` | ~43 GB | Best quality of these, needs a big machine |

Avoid `phi4`: its 16K context is too small for comfortable chat. Bigger is usually
more faithful, so choose the largest model your hardware runs smoothly. Compare
models on a video you know well; quality differs by content. After changing the
model, restart the server (and re-pull in Docker via `ollama-pull`).


## Limitations

- **Only English captions** are supported (auto-generated captions are fine).
- Retrieval finds the few most relevant passages, so **very broad questions**
  ("summarise the whole video") work worse than specific ones. Raising
  `RETRIEVER_K` helps at the cost of more context.
- Video indexes live in server memory; they are rebuilt automatically when needed.
- Chat history is per browser and stored unencrypted in `localStorage`; don't use
  shared computers for private videos.
- The 3-chat limit is enforced by the browser, not by an account system.
- Token counts are estimates, not exact.

## Security notes

- Never commit `.env`; it is git-ignored and excluded from the Docker image.
- A key entered in the UI is held in `sessionStorage` and sent with requests; the server does not store it.