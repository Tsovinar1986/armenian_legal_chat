// Wires the console UI to the real FastAPI backend (api.py) instead of the
// scripted demo in docs/legal-ui.html. Vite's dev-server proxy (vite.config.js)
// forwards /api and /health to http://localhost:8000, so every call here is a
// same-origin relative path — no CORS setup needed on the backend.

const transcript = document.getElementById("transcript");
const typedInput = document.getElementById("typedInput");
const btnSend = document.getElementById("btnSend");
const btnMic = document.getElementById("btnMic");
const btnUpload = document.getElementById("btnUpload");
const fileInput = document.getElementById("fileInput");
const consoleDot = document.getElementById("liveDot");
const backendPill = document.getElementById("backendPill");
const backendStatus = document.getElementById("backendStatus");
const langButtons = document.querySelectorAll(".lang-btn");
const greetingBubble = document.getElementById("greetingBubble");
const consoleTitleText = document.getElementById("consoleTitleText");
const onboarding = document.getElementById("onboarding");
const onboardingClose = document.getElementById("onboardingClose");

const MAX_RECORDING_MS = 12000;
const ONBOARDING_DISMISSED_KEY = "legalui.onboardingDismissed";

if (onboarding && onboardingClose) {
  if (localStorage.getItem(ONBOARDING_DISMISSED_KEY) === "1") {
    onboarding.hidden = true;
  }
  onboardingClose.addEventListener("click", () => {
    onboarding.hidden = true;
    localStorage.setItem(ONBOARDING_DISMISSED_KEY, "1");
  });
}

// Matches api.py's STT_LANGUAGE_MAP / src/agents/legal_crew.py's LANGUAGE_NAMES —
// the same three short codes drive both the chat response language and the
// mic's speech-recognition locale.
const PLACEHOLDERS = {
  hy: "Տվեք ձեր իրավական հարցը հայերեն…",
  en: "Type your legal question in English…",
  ru: "Введите ваш юридический вопрос на русском…",
};
const GREETINGS = {
  hy: "🤖 Ինչպե՞ս կարող եմ օգնել ձեզ այսօր...",
  en: "🤖 How can I help you today...",
  ru: "🤖 Чем я могу вам помочь сегодня...",
};
const CONSOLE_TITLES = {
  hy: "Հայկական իրավական օգնական",
  en: "Armenian Legal Assistant",
  ru: "Армянский юридический помощник",
};

let sessionId = null;
let busy = false;
let mediaRecorder = null;
let recordingTimer = null;
let currentLanguage = "hy";

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
  bubble.className = `bubble${opts.warn ? " warn" : ""}${opts.videoEmbed ? " video-embed" : ""}`;
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

function setLanguage(lang) {
  if (!PLACEHOLDERS[lang] || lang === currentLanguage) return;
  currentLanguage = lang;
  langButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.lang === lang));
  typedInput.placeholder = PLACEHOLDERS[lang];
  // Only the static opening greeting follows the language switch in place —
  // real exchanges already in the transcript stay exactly as they were sent
  // and answered, and switching languages doesn't spam the chat with a
  // separate notice for every click.
  if (greetingBubble) greetingBubble.textContent = GREETINGS[lang];
  if (consoleTitleText) consoleTitleText.textContent = CONSOLE_TITLES[lang];
}

langButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (busy) return;
    setLanguage(btn.dataset.lang);
  });
});

async function sendMessage(message) {
  addRow("user", "You", message);
  setComposerEnabled(false);
  const typing = addTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, language: currentLanguage }),
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

// ---------- Mic: record with MediaRecorder, transcribe via /api/speech-to-text ----------

function pickRecorderMime() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    addRow("system", null, "🎙️ This browser doesn't support in-page audio recording.", { warn: true });
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    addRow("system", null, `🎙️ Microphone permission denied (${err.message}).`, { warn: true });
    return;
  }

  const mimeType = pickRecorderMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];

  mediaRecorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  });

  mediaRecorder.addEventListener("stop", () => {
    stream.getTracks().forEach((t) => t.stop());
    clearTimeout(recordingTimer);
    btnMic.classList.remove("active");
    btnMic.innerHTML = '<kbd>m</kbd>🎤 Ask by voice';
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
    mediaRecorder = null;
    transcribeAndSend(blob);
  });

  mediaRecorder.start();
  btnMic.classList.add("active");
  btnMic.innerHTML = '<kbd>m</kbd>⏺ Recording… click to stop';
  addRow("system", null, "🎤 Listening… speak now");
  recordingTimer = setTimeout(() => stopRecording(), MAX_RECORDING_MS);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function transcribeAndSend(blob) {
  setComposerEnabled(false);
  const working = addRow("system", null, "📝 Transcribing…");

  try {
    const form = new FormData();
    const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
    form.append("file", blob, `mic-input.${ext}`);
    form.append("language", currentLanguage);

    const res = await fetch("/api/speech-to-text", { method: "POST", body: form });
    const data = await res.json();
    working.row.remove();

    if (!res.ok || data.success === false) {
      addRow("system", null, data.message || `HTTP ${res.status}`, { warn: true });
      setComposerEnabled(true);
      return;
    }

    setComposerEnabled(true);
    sendMessage(data.text);
  } catch (err) {
    working.row.remove();
    addRow("system", null, `⚠️ Speech-to-text request failed: ${err.message}`, { warn: true });
    setComposerEnabled(true);
  }
}

btnMic.addEventListener("click", () => {
  if (busy) return;
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    startRecording();
  }
});

// ---------- Upload: send file straight to /api/upload ----------

btnUpload.addEventListener("click", () => {
  if (busy) return;
  fileInput.click();
});

fileInput.addEventListener("change", async () => {
  const f = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!f) return;

  const isVideo = /\.(mp4|mov|avi|mkv)$/i.test(f.name);
  setComposerEnabled(false);

  // Play the actual uploaded video right in the console — analysis results
  // (actions/emotion) land as a follow-up message below it, so you can watch
  // what was uploaded while reading what the backend found in it. This is a
  // local object URL (the file never left the browser for this), separate
  // from the copy POSTed to /api/upload for analysis below.
  if (isVideo) {
    const objectUrl = URL.createObjectURL(f);
    addRow(
      "system", null,
      `<video controls playsinline preload="metadata" src="${objectUrl}"></video>`,
      { html: true, videoEmbed: true }
    );
  }

  const working = addRow(
    "system", null,
    isVideo
      ? `🎥 Analyzing ${f.name}… (loads YOLO/MediaPipe on first upload, can take a while)`
      : `📄 Processing ${f.name}…`
  );

  try {
    const form = new FormData();
    form.append("file", f);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    working.row.remove();

    if (!res.ok || data.success === false) {
      addRow("system", null, data.message || `HTTP ${res.status}`, { warn: true });
    } else if (data.kind === "video") {
      const actions = data.actions && data.actions.length ? data.actions.join(", ") : "none detected";
      addRow(
        "system", null,
        `✅ Analyzed ${data.frames_analyzed} sampled frame(s). Actions: ${actions}. Emotion: ${data.emotion || "n/a"}.`
      );
    } else {
      addRow("system", null, `✅ ${data.message}`);
    }
  } catch (err) {
    working.row.remove();
    addRow("system", null, `⚠️ Upload failed: ${err.message}`, { warn: true });
  } finally {
    setComposerEnabled(true);
  }
});

document.addEventListener("keydown", (e) => {
  if (document.activeElement === typedInput) return;
  if (e.key === "t") typedInput.focus();
  if (e.key === "m") btnMic.click();
  if (e.key === "u") btnUpload.click();
});

typedInput.placeholder = PLACEHOLDERS[currentLanguage];
checkBackend();
setInterval(checkBackend, 15000);
