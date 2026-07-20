// Wires the console UI to the real FastAPI backend (api.py) instead of the
// scripted demo in docs/legal-ui.html. Vite's dev-server proxy (vite.config.js)
// forwards /api and /health to http://localhost:8000, so every call here is a
// same-origin relative path — no CORS setup needed on the backend.

const transcript = document.getElementById("transcript");
const typedInput = document.getElementById("typedInput");
const btnSend = document.getElementById("btnSend");
const btnMic = document.getElementById("btnMic");
const btnUpload = document.getElementById("btnUpload");
const consoleDot = document.getElementById("liveDot");
const backendPill = document.getElementById("backendPill");
const backendStatus = document.getElementById("backendStatus");

let sessionId = null;
let busy = false;

function scrollToEnd() {
  transcript.scrollTop = transcript.scrollHeight;
}

function addRow(kind, who, text, opts = {}) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;

  if (who) {
    const whoEl = document.createElement("span");
    whoEl.className = "who";
    whoEl.textContent = who;
    row.appendChild(whoEl);
  }

  const bubble = document.createElement("div");
  bubble.className = `bubble${opts.warn ? " warn" : ""}`;
  if (opts.html) {
    bubble.innerHTML = text;
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);

  transcript.appendChild(row);
  scrollToEnd();
  return { row, bubble };
}

function addTyping() {
  return addRow("bot", "AI", '<span class="typing"><span></span><span></span><span></span></span>', { html: true });
}

function setComposerEnabled(on) {
  busy = !on;
  typedInput.disabled = !on;
  btnSend.disabled = !on;
}

function setBackendStatus(online) {
  consoleDot.classList.toggle("online", online);
  consoleDot.classList.toggle("offline", !online);
  backendPill.classList.toggle("online", online);
  backendPill.classList.toggle("offline", !online);
  backendStatus.textContent = online ? "backend online" : "backend unreachable";
}

async function checkBackend() {
  try {
    const res = await fetch("/health");
    setBackendStatus(res.ok);
  } catch {
    setBackendStatus(false);
  }
}

async function sendMessage(message) {
  addRow("user", "You", message);
  setComposerEnabled(false);
  const typing = addTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, language: "hy" }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id || sessionId;
    typing.row.remove();

    if (data.success === false) {
      addRow("system", null, data.message || "The backend rejected that message.", { warn: true });
    } else {
      addRow("bot", "AI", data.response || "(empty response)");
    }
    setBackendStatus(true);
  } catch (err) {
    typing.row.remove();
    addRow(
      "system",
      null,
      `⚠️ Couldn't reach the backend (${err.message}). Is "uvicorn api:app --reload --port 8000" running?`,
      { warn: true }
    );
    setBackendStatus(false);
  } finally {
    setComposerEnabled(true);
    typedInput.focus();
  }
}

function submitTyped() {
  if (busy) return;
  const val = typedInput.value.trim();
  if (!val) return;
  typedInput.value = "";
  sendMessage(val);
}

btnSend.addEventListener("click", submitTyped);
typedInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitTyped();
});

btnMic.addEventListener("click", () => {
  addRow(
    "system",
    null,
    "🎙️ Voice input isn't exposed over HTTP — it only runs in the desktop CLI (src/main.py, press [m]/[v] there).",
    { warn: true }
  );
});

btnUpload.addEventListener("click", () => {
  addRow(
    "system",
    null,
    "📎 Upload isn't exposed over HTTP — it only runs in the desktop CLI (src/main.py, press [u] there).",
    { warn: true }
  );
});

document.addEventListener("keydown", (e) => {
  if (document.activeElement === typedInput) return;
  if (e.key === "t") typedInput.focus();
  if (e.key === "m") btnMic.click();
  if (e.key === "u") btnUpload.click();
});

checkBackend();
setInterval(checkBackend, 15000);
