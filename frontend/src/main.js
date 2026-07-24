// Wires the console UI to the real FastAPI backend (api.py). Vite's
// dev-server proxy (vite.config.js) forwards /api and /health to
// http://localhost:8000, so every call here is a same-origin relative path.

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

// Was 12s — too short for anything but a one-line question and would
// silently cut off longer sentences mid-word (nothing past the cap is ever
// sent, since the recorder just stops and hands over whatever it has).
const MAX_RECORDING_MS = 60000;

// Matches api.py's STT_LANGUAGE_MAP — the same three short codes drive both
// the chat response language and the mic/video speech-recognition locale.
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

function addRow(kind, who, content, opts = {}) {
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
    bubble.innerHTML = content;
  } else {
    bubble.textContent = content;
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
  btnMic.disabled = !on;
  btnUpload.disabled = !on;
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
checkBackend();

function setLanguage(lang) {
  if (!PLACEHOLDERS[lang] || lang === currentLanguage) return;
  currentLanguage = lang;
  langButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.lang === lang));
  typedInput.placeholder = PLACEHOLDERS[lang];
  if (greetingBubble) greetingBubble.textContent = GREETINGS[lang];
  if (consoleTitleText) consoleTitleText.textContent = CONSOLE_TITLES[lang];
}

langButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (busy) return;
    setLanguage(btn.dataset.lang);
  });
});

// ---------- Chat: typed or transcribed text -> /api/chat ----------

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

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

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
      "system", null,
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

// ---------- Upload: doc -> case DB, video -> action/emotion + speech transcript ----------

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

  // Play the actual uploaded video right in the console so you can watch it
  // while reading the analysis + transcript that lands as a chat reply
  // below. This is a local object URL — the file itself is separately
  // POSTed to /api/upload for analysis.
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
      ? `🎥 Analyzing ${f.name}… (action/emotion detection + speech transcript, can take a while on first upload)`
      : `📄 Processing ${f.name}…`
  );

  try {
    const form = new FormData();
    form.append("file", f);
    form.append("language", currentLanguage);
    if (sessionId) form.append("session_id", sessionId);

    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    working.row.remove();

    if (!res.ok || data.success === false) {
      addRow("system", null, data.message || `HTTP ${res.status}`, { warn: true });
    } else if (data.kind === "video") {
      const actions = data.actions && data.actions.length ? data.actions.join(", ") : "none detected";
      const spoken = data.transcript
        ? `<br><strong>🗣️ What was said:</strong> "${data.transcript}"`
        : `<br><strong>🗣️ What was said:</strong> no speech detected — based on action detection, here's what's happening: ${actions}`;
      const emotionChanges = data.emotion_changes && data.emotion_changes.length > 1
        ? `<br><strong>Emotion changed during video:</strong> ${data.emotion_changes.join(" → ")}`
        : "";
      addRow(
        "bot", "AI",
        `✅ Analyzed ${data.frames_analyzed} sampled frame(s).<br>` +
        `<strong>Actions:</strong> ${actions}<br>` +
        `<strong>Emotion:</strong> ${data.emotion || "n/a"}` +
        emotionChanges +
        spoken,
        { html: true }
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
