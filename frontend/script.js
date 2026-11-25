/**
 * script.js - frontend logic for YouTube Chat Boat.                    (REWRITTEN)
 *
 * What changed compared with the previous version
 * ------------------------------------------------
 *  - NEW  Up to 3 chats are kept (see MAX_CHATS in backend/config.py). Each chat
 *         = one video + its whole conversation. They are saved in this browser's
 *         localStorage, so they survive a page reload or server restart.
 *  - NEW  Each question is sent together with the recent conversation, so
 *         follow-up questions work. The server trims it to the model's 10k-token
 *         context window (see backend/rag/context.py).
 *  - NEW  A small meter shows how much of the context window the last answer used.
 *  - NEW  If the server forgot a video (restart / eviction), the chat silently
 *         re-indexes it from the saved link and carries on.
 *  - NEW  ChatGPT-style layout: sidebar, messages, message box.
 *  - KEPT the load -> ask flow, Ollama/OpenAI choice, key held in sessionStorage,
 *         and timestamp chips that open the video at the right moment.
 *
 * Code map:  1 Settings & state | 2 Small helpers | 3 Saving chats | 4 Talking to
 *            the server | 5 Drawing the screen | 6 Actions | 7 Event wiring | 8 Start-up
 */
"use strict";

/* ========================================================================
 * 1. SETTINGS & STATE
 * ====================================================================== */
const $ = (id) => document.getElementById(id);

const STORAGE_KEY = "ytchat.v1";   // localStorage key holding all chats
const HISTORY_SENT = 40;           // most recent messages sent with each question

/** Defaults; overwritten by GET /api/config at start-up. */
let config = {
  default_provider: "ollama", server_openai_key: false,
  max_chats: 3, context_window: 10000, ollama_model: "", openai_model: "",
};

/**
 * All saved chats. A chat looks like:
 * { id, title, url, videoId, provider, sessionId, createdAt, updatedAt,
 *   messages: [{ role: "user"|"assistant", content, sources?: [seconds], usage? }] }
 */
const store = { chats: [], activeId: null };

const busy = new Set();      // ids of chats with a question being answered right now
const transient = new Map(); // chatId -> { kind: "pending"|"error", text } (never saved)

/* ========================================================================
 * 2. SMALL HELPERS
 * ====================================================================== */
const providerValue = () => document.querySelector('input[name="provider"]:checked').value;
const activeChat = () => store.chats.find((c) => c.id === store.activeId) || null;
/** Most recently used chat first. */
const sortedChats = () => [...store.chats].sort((a, b) => b.updatedAt - a.updatedAt);
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));

/** The OpenAI key lives only in sessionStorage (cleared when the tab closes). */
const getKey = () => { try { return sessionStorage.getItem("openai_key") || ""; } catch { return ""; } };
const setKey = (v) => { try { sessionStorage.setItem("openai_key", v); } catch { /* storage blocked */ } };

/** 187 -> "3:07", 3725 -> "1:02:05" (same format as the server). */
function fmtTime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}
/** 8500 -> "8.5k", 10000 -> "10k" */
const fmtTokens = (n) => (n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n));

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

/** Grow the message box with its text, up to a maximum height. */
function autosize() {
  const box = $("question");
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 200) + "px";
}

/* ========================================================================
 * 3. SAVING CHATS (localStorage)
 * ====================================================================== */

/** Accept only well-formed chats, so corrupted or hand-edited data cannot break the page. */
function cleanChat(c) {
  if (!c || typeof c.id !== "string" || typeof c.url !== "string") return null;
  if (!/^[\w-]{11}$/.test(c.videoId || "")) return null;
  if (!["ollama", "openai"].includes(c.provider)) return null;
  const messages = (Array.isArray(c.messages) ? c.messages : [])
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .map((m) => ({
      role: m.role, content: m.content,
      sources: Array.isArray(m.sources) ? m.sources.filter(Number.isFinite) : [],
      usage: m.usage && Number.isFinite(m.usage.used) ? m.usage : undefined,
    }));
  return {
    id: c.id, url: c.url, videoId: c.videoId, provider: c.provider,
    title: String(c.title || `Video ${c.videoId}`), sessionId: String(c.sessionId || ""),
    createdAt: Number(c.createdAt) || Date.now(), updatedAt: Number(c.updatedAt) || Date.now(), messages,
  };
}

function loadStore() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    store.chats = (Array.isArray(data.chats) ? data.chats : []).map(cleanChat).filter(Boolean);
    store.activeId = store.chats.some((c) => c.id === data.activeId) ? data.activeId : null;
    // Opening the page with saved chats shows the most recent one.
    if (!store.activeId && store.chats.length) store.activeId = sortedChats()[0].id;
  } catch { /* nothing saved yet, or unreadable: start empty */ }
}

function saveStore() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ chats: store.chats, activeId: store.activeId }));
  } catch { /* storage full or blocked: the app still works, just without saving */ }
}

/* ========================================================================
 * 4. TALKING TO THE SERVER
 * ====================================================================== */

/** Call the JSON API. Errors carry the server's message and the HTTP `status`. */
async function api(path, body, method = "POST") {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("Could not reach the server. Is it running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ========================================================================
 * 5. DRAWING THE SCREEN
 * ====================================================================== */

/** Sidebar: the list of saved chats. */
function renderSidebar() {
  const list = $("chat-list");
  list.replaceChildren();
  for (const chat of sortedChats()) {
    const li = document.createElement("li");
    li.className = "chat-item" + (chat.id === store.activeId ? " active" : "");

    const open = document.createElement("button");
    open.type = "button";
    open.className = "chat-open";
    open.title = chat.title;
    if (chat.id === store.activeId) open.setAttribute("aria-current", "true");
    const name = document.createElement("span");
    name.className = "chat-name";
    name.textContent = chat.title;
    const sub = document.createElement("span");
    sub.className = "chat-sub";
    const asked = chat.messages.filter((m) => m.role === "user").length;
    sub.textContent = `${asked} question${asked === 1 ? "" : "s"} \u00b7 ${chat.provider === "openai" ? "OpenAI" : "Ollama"}`;
    open.append(name, sub);
    open.addEventListener("click", () => openChat(chat.id));

    const del = document.createElement("button");
    del.type = "button";
    del.className = "chat-del";
    del.setAttribute("aria-label", `Delete chat: ${chat.title}`);
    del.textContent = "\u00d7";
    del.addEventListener("click", () => deleteChat(chat.id));

    li.append(open, del);
    list.append(li);
  }
  $("chat-count").textContent = `(${store.chats.length}/${config.max_chats})`;
  $("side-empty").hidden = store.chats.length > 0;
  $("side-note").textContent = `Only your ${config.max_chats} most recent chats are kept, in this browser.`;
}

/** Start screen: warn when a new chat would push the oldest one out. */
function renderStart() {
  const note = $("limit-note");
  if (store.chats.length >= config.max_chats) {
    const oldest = sortedChats().at(-1);
    note.textContent = `You already have ${store.chats.length} chats. Starting a new one removes the oldest: \u201c${oldest.title}\u201d.`;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

/** A bot-side avatar (the little red play button). */
function avatar() {
  const a = document.createElement("div");
  a.className = "avatar";
  a.setAttribute("aria-hidden", "true");
  a.textContent = "\u25b6";
  return a;
}

/** One saved message -> an <li>. `showNote` adds the "older messages dropped" hint. */
function messageEl(chat, m, showNote) {
  const li = document.createElement("li");
  if (m.role === "user") {
    li.className = "msg user";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = m.content;      // textContent: never interpreted as HTML
    li.append(body);
    return li;
  }

  li.className = "msg bot";
  const body = document.createElement("div");
  body.className = "body";
  const text = document.createElement("div");
  text.className = "text";
  text.innerHTML = renderMarkdown(m.content, { videoId: chat.videoId }); // escaped inside markdown.js
  body.append(text);

  if (m.sources && m.sources.length) {  // chips that jump to the moment in the video
    const wrap = document.createElement("div");
    wrap.className = "stamps";
    for (const s of m.sources) {
      const a = document.createElement("a");
      a.href = `https://www.youtube.com/watch?v=${chat.videoId}&t=${s}s`;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = fmtTime(s);
      wrap.append(a);
    }
    body.append(wrap);
  }
  if (showNote && m.usage && m.usage.dropped > 0) {
    const note = document.createElement("p");
    note.className = "drop-note";
    note.textContent = `The model's ${fmtTokens(config.context_window)}-token memory is full, so the ${m.usage.dropped} oldest message${m.usage.dropped === 1 ? "" : "s"} are no longer used for answers. They stay visible here.`;
    body.append(note);
  }
  li.append(avatar(), body);
  return li;
}

/** The animated "thinking" row (or a status like "Reconnecting..."). */
function pendingEl(text) {
  const li = document.createElement("li");
  li.className = "msg bot";
  const body = document.createElement("div");
  body.className = "body";
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  const label = document.createElement("span");
  label.textContent = text || "Thinking\u2026";
  typing.append(label);
  body.append(typing);
  li.append(avatar(), body);
  return li;
}

/** An error row shown in place of an answer. */
function errorEl(text) {
  const li = document.createElement("li");
  li.className = "msg bot error";
  const body = document.createElement("div");
  body.className = "body";
  const t = document.createElement("div");
  t.className = "text";
  t.textContent = text;
  body.append(t);
  li.append(avatar(), body);
  return li;
}

/** Conversation of the open chat. */
function renderMessages() {
  const chat = activeChat();
  if (!chat) return;
  const log = $("log");
  log.replaceChildren();

  const pending = transient.get(chat.id);
  if (!chat.messages.length && !pending) {
    const hint = document.createElement("li");
    hint.className = "hint";
    hint.textContent = "Ask a question about this video. Answers link to the moment they came from.";
    log.append(hint);
  }
  const lastBot = chat.messages.map((m) => m.role).lastIndexOf("assistant");
  chat.messages.forEach((m, i) => log.append(messageEl(chat, m, i === lastBot)));
  if (pending) log.append(pending.kind === "pending" ? pendingEl(pending.text) : errorEl(pending.text));

  $("scroll").scrollTop = $("scroll").scrollHeight;
}

/** Context meter in the top bar: how full the model's memory was on the last answer. */
function renderMeter() {
  const chat = activeChat();
  const last = chat && [...chat.messages].reverse().find((m) => m.role === "assistant" && m.usage);
  $("meter").hidden = !last;
  if (!last) return;
  const { used, limit, dropped } = last.usage;
  const pct = Math.min(100, Math.round((used / limit) * 100));
  $("meter-fill").style.width = pct + "%";
  $("meter-text").textContent = `~${fmtTokens(used)} / ${fmtTokens(limit)}`;
  $("meter").classList.toggle("warn", dropped > 0 || pct >= 80);
  $("meter").title =
    `Estimated context used by the last answer: about ${used.toLocaleString()} of ${limit.toLocaleString()} tokens ` +
    `(part of the window is kept free for the reply).` +
    (dropped ? ` ${dropped} older message(s) no longer fit.` : "");
}

/** Disable "Send" while the open chat is waiting for an answer. */
function updateComposerState() {
  const chat = activeChat();
  $("ask-btn").disabled = !chat || busy.has(chat.id);
}

/** Switch between the start screen and the conversation. */
function renderMain() {
  const chat = activeChat();
  $("start").hidden = !!chat;
  $("log").hidden = !chat;
  $("composer-wrap").hidden = !chat;
  $("video-toggle").hidden = !chat;
  $("chat-title").textContent = chat ? chat.title : "New chat";
  document.title = chat ? `${chat.title} \u00b7 YouTube Chat Boat` : "YouTube Chat Boat";
  if (chat) renderMessages(); else renderStart();
  renderMeter();
  updateComposerState();
}

function renderAll() { renderSidebar(); renderMain(); }

/** Redraw only if `chat` is the one on screen (used after slow requests finish). */
function renderIfActive(chat) {
  renderSidebar();
  if (store.activeId === chat.id) {
    renderMessages(); renderMeter(); updateComposerState();
  }
}

/* ---- video player (hidden until "Watch video" is pressed) ---- */
function openPlayer(chat) {
  const frame = document.createElement("iframe");
  frame.src = `https://www.youtube.com/embed/${chat.videoId}`;
  frame.title = "Video player";
  frame.allowFullscreen = true;
  $("player").replaceChildren(frame);
  $("player").hidden = false;
  $("video-toggle").textContent = "Hide video";
  $("video-toggle").setAttribute("aria-expanded", "true");
}
function closePlayer() {
  $("player").replaceChildren();   // removing the iframe also stops playback
  $("player").hidden = true;
  $("video-toggle").textContent = "Watch video";
  $("video-toggle").setAttribute("aria-expanded", "false");
}

/* ---- sidebar on small screens ---- */
function openSidebar() { $("app").classList.add("open"); $("scrim").hidden = false; }
function closeSidebar() { $("app").classList.remove("open"); $("scrim").hidden = true; }

/* ========================================================================
 * 6. ACTIONS
 * ====================================================================== */

function openChat(id) {
  store.activeId = id;
  saveStore();
  closePlayer();
  closeSidebar();
  renderAll();
  $("question").focus();
}

function newChat() {
  store.activeId = null;
  saveStore();
  closePlayer();
  closeSidebar();
  setStatus("");
  renderAll();
  $("url").focus();
}

/** Remove a chat and tell the server to free its search index. */
function removeChat(id) {
  const chat = store.chats.find((c) => c.id === id);
  if (!chat) return;
  store.chats = store.chats.filter((c) => c.id !== id);
  transient.delete(id);
  if (chat.sessionId) {
    // Fire and forget: failure only means the server frees the memory later.
    fetch(`/api/session/${encodeURIComponent(chat.sessionId)}`, { method: "DELETE", keepalive: true }).catch(() => {});
  }
  if (store.activeId === id) {
    store.activeId = sortedChats()[0]?.id ?? null;
    closePlayer();
  }
}

function deleteChat(id) {
  const chat = store.chats.find((c) => c.id === id);
  if (!chat || !confirm(`Delete this chat?\n\n${chat.title}`)) return;
  removeChat(id);
  saveStore();
  renderAll();
}

/** Keep at most `max_chats` chats by dropping the least recently used ones. */
function enforceLimit(keepId) {
  while (store.chats.length > config.max_chats) {
    const oldest = sortedChats().filter((c) => c.id !== keepId).at(-1);
    if (!oldest) break;
    removeChat(oldest.id);
  }
}

/** Start screen submit: index the video and open a brand-new chat. */
async function startChat(e) {
  e.preventDefault();
  $("load-btn").disabled = true;
  setStatus("Fetching captions and indexing. The first run can take a minute.");
  try {
    const url = $("url").value.trim();
    const provider = providerValue();
    const data = await api("/api/load", { url, provider, api_key: getKey() });

    const now = Date.now();
    const chat = {
      id: uid(), url, provider, videoId: data.video_id, sessionId: data.session_id,
      title: data.title || `Video ${data.video_id}`, createdAt: now, updatedAt: now, messages: [],
    };
    store.chats.push(chat);
    store.activeId = chat.id;
    enforceLimit(chat.id);          // the 4th chat pushes the oldest one out
    saveStore();

    $("url").value = "";
    setStatus("");
    closePlayer();
    renderAll();
    $("question").focus();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("load-btn").disabled = false;
  }
}

/** Ask the server for an answer, re-indexing the video once if the server forgot it. */
async function requestAnswer(chat, question, history) {
  const send = () => api("/api/chat", { session_id: chat.sessionId, question, history, api_key: getKey() });
  try {
    return await send();
  } catch (err) {
    if (err.status !== 404) throw err;
    // The server lost the index (restart or eviction): rebuild it from the saved link.
    transient.set(chat.id, { kind: "pending", text: "Reconnecting to the video\u2026" });
    renderIfActive(chat);
    const data = await api("/api/load", { url: chat.url, provider: chat.provider, api_key: getKey() });
    chat.sessionId = data.session_id;
    transient.set(chat.id, { kind: "pending", text: "" });
    renderIfActive(chat);
    return await send();
  }
}

/** Send one question and show the answer. */
async function ask(chat, question) {
  // What the model should remember: the conversation BEFORE this question.
  const history = chat.messages.slice(-HISTORY_SENT).map(({ role, content }) => ({ role, content }));

  chat.messages.push({ role: "user", content: question });
  chat.updatedAt = Date.now();
  busy.add(chat.id);
  transient.set(chat.id, { kind: "pending", text: "" });
  saveStore();
  renderIfActive(chat);

  try {
    const data = await requestAnswer(chat, question, history);
    chat.messages.push({ role: "assistant", content: data.answer, sources: data.sources || [], usage: data.usage });
    transient.delete(chat.id);
  } catch (err) {
    // Take the failed question back out so the saved history stays clean,
    // and put the text back in the box so the user can simply press Send again.
    chat.messages.pop();
    transient.set(chat.id, { kind: "error", text: err.message });
    if (store.activeId === chat.id && !$("question").value) { $("question").value = question; autosize(); }
  } finally {
    busy.delete(chat.id);
    if (store.chats.includes(chat)) {   // the chat may have been deleted meanwhile
      chat.updatedAt = Date.now();
      saveStore();
      renderIfActive(chat);
    }
    if (store.activeId === chat.id) $("question").focus();
  }
}

/* ========================================================================
 * 7. EVENT WIRING
 * ====================================================================== */
$("new-chat").addEventListener("click", newChat);
$("load-form").addEventListener("submit", startChat);
$("menu-btn").addEventListener("click", openSidebar);
$("scrim").addEventListener("click", closeSidebar);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });

$("video-toggle").addEventListener("click", () => {
  const chat = activeChat();
  if (!chat) return;
  if ($("player").hidden) openPlayer(chat); else closePlayer();
});

// The OpenAI key box only appears for OpenAI. The key is remembered per tab.
document.querySelectorAll('input[name="provider"]').forEach((r) =>
  r.addEventListener("change", () => ($("key-box").hidden = providerValue() !== "openai"))
);
$("key").value = getKey();
$("key").addEventListener("input", () => setKey($("key").value));

// Message box: Enter sends, Shift+Enter adds a new line.
$("question").addEventListener("input", autosize);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const chat = activeChat();
  const question = $("question").value.trim();
  if (!chat || !question || busy.has(chat.id)) return;
  $("question").value = "";
  autosize();
  ask(chat, question);
});

/* ========================================================================
 * 8. START-UP
 * ====================================================================== */

/** Apply the server's settings to the start screen. */
function applyConfig(cfg) {
  config = { ...config, ...cfg };
  document.querySelector(`input[name="provider"][value="${config.default_provider}"]`)?.click();
  if (config.ollama_model) $("ollama-name").textContent = `${config.ollama_model} \u00b7 local & free`;
  if (config.openai_model) {
    $("openai-name").textContent = `${config.openai_model} \u00b7 ` + (config.server_openai_key ? "key provided by server" : "needs your key");
  }
  if (config.server_openai_key) $("key-note").textContent = "This server has its own key, so you can leave this empty.";
}

loadStore();
fetch("/api/config")
  .then((r) => r.json())
  .then(applyConfig)
  .catch(() => { /* server unreachable: keep defaults */ })
  .finally(() => {
    enforceLimit(store.activeId);   // in case MAX_CHATS was lowered since the last visit
    saveStore();
    renderAll();
    if (!activeChat()) $("url").focus();
  });
