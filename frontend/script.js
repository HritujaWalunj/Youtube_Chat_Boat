const $ = (id) => document.getElementById(id);
const state = { sessionId: null, videoId: null };

const providerValue = () => document.querySelector('input[name="provider"]:checked').value;
const setStatus = (msg, isError = false) => {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
};

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

function addMessage(kind, text, sources = []) {
  const hint = document.querySelector("#log .hint");
  if (hint) hint.remove();
  const li = document.createElement("li");
  li.className = kind;
  li.textContent = text;
  if (sources.length) {
    const wrap = document.createElement("div");
    wrap.className = "stamps";
    for (const s of sources) {
      const a = document.createElement("a");
      a.href = `https://www.youtube.com/watch?v=${state.videoId}&t=${s}s`;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
      wrap.append(a);
    }
    li.append(wrap);
  }
  $("log").append(li);
  $("log").scrollTop = $("log").scrollHeight;
  return li;
}

// Provider toggle: the key box only appears for OpenAI.
document.querySelectorAll('input[name="provider"]').forEach((r) =>
  r.addEventListener("change", () => ($("key-box").hidden = providerValue() !== "openai"))
);

try { $("key").value = sessionStorage.getItem("openai_key") || ""; } catch {}
$("key").addEventListener("input", () => {
  try { sessionStorage.setItem("openai_key", $("key").value); } catch {}
});

fetch("/api/config").then((r) => r.json()).then((cfg) => {
  document.querySelector(`input[value="${cfg.default_provider}"]`)?.click();
  if (cfg.server_openai_key)
    $("key-note").textContent = "This server has its own key, so you can leave this empty.";
}).catch(() => {});

$("load-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("load-btn").disabled = true;
  setStatus("Fetching captions and indexing. The first run can take a minute.");
  try {
    const data = await api("/api/load", {
      url: $("url").value,
      provider: providerValue(),
      api_key: $("key").value,
    });
    state.sessionId = data.session_id;
    state.videoId = data.video_id;
    $("player").hidden = false;
    $("player").innerHTML = "";
    const frame = document.createElement("iframe");
    frame.src = `https://www.youtube.com/embed/${data.video_id}`;
    frame.title = "Video player";
    frame.allowFullscreen = true;
    $("player").append(frame);
    $("log").innerHTML = "";
    $("question").disabled = $("ask-btn").disabled = false;
    $("question").focus();
    setStatus(`Ready. Indexed ${data.chunks} sections.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("load-btn").disabled = false;
  }
});

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  addMessage("user", question);
  const pending = addMessage("bot", "Thinking...");
  $("ask-btn").disabled = true;
  try {
    const data = await api("/api/chat", {
      session_id: state.sessionId, question, api_key: $("key").value,
    });
    pending.remove();
    addMessage("bot", data.answer, data.sources);
  } catch (err) {
    pending.textContent = err.message;
    pending.classList.add("error");
  } finally {
    $("ask-btn").disabled = false;
    $("question").focus();
  }
});