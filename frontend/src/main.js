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

// ---------- Account: sign in / register / forgot password, via /api/auth/* ----------
// Same backend endpoints as api.py's own portal UI (src/db/portal_store.py),
// reached here through Vite's /api proxy — see vite.config.js.

const AUTH_TOKEN_KEY = "legalui.authToken";
const AUTH_USER_KEY = "legalui.authUser";
const ROLE_LABELS = { individual: "Individual", lawyer: "Lawyer", therapist: "Therapist" };

const accountPill = document.getElementById("accountPill");
const accountStatus = document.getElementById("accountStatus");
const authOverlay = document.getElementById("authOverlay");
const authClose = document.getElementById("authClose");
const authMessage = document.getElementById("authMessage");
const roleToggle = document.getElementById("roleToggle");
const authTabs = document.getElementById("authTabs");
const panelAccount = document.getElementById("panelAccount");
const roleToggleButtons = document.querySelectorAll("#roleToggle button");
const roleLabelEls = document.querySelectorAll(".roleLabel");
const paymentsLink = document.getElementById("paymentsLink");

let selectedRole = "individual";
let authToken = localStorage.getItem(AUTH_TOKEN_KEY) || null;
let authUser = null;
try {
  authUser = JSON.parse(localStorage.getItem(AUTH_USER_KEY) || "null");
} catch {
  authUser = null;
}

function updateAccountPill() {
  accountStatus.textContent = authUser ? `${authUser.name} · ${ROLE_LABELS[authUser.role] || authUser.role}` : "Sign in";
  accountPill.classList.toggle("signed-in", !!authUser);
}

const authTabButtons = document.querySelectorAll("#authTabs button");
const authPanels = {
  quick: document.getElementById("panelQuick"),
  signin: document.getElementById("panelSignin"),
  register: document.getElementById("panelRegister"),
  payments: document.getElementById("panelPayments"),
};

function setAuthTab(tab) {
  authTabButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
  Object.entries(authPanels).forEach(([key, panel]) => { panel.hidden = key !== tab; });
}
authTabButtons.forEach((btn) => btn.addEventListener("click", () => setAuthTab(btn.dataset.tab)));

function setRole(role) {
  selectedRole = role;
  roleToggleButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.role === role));
  roleLabelEls.forEach((el) => { el.textContent = ROLE_LABELS[role]; });
  paymentsLink.href = `/pay?consultation_type=${role === "therapist" ? "therapist" : "lawyer"}`;
}
roleToggleButtons.forEach((btn) => btn.addEventListener("click", () => setRole(btn.dataset.role)));

function openAuthModal() {
  authMessage.textContent = "";
  authOverlay.hidden = false;
  if (authUser) {
    roleToggle.hidden = true;
    authTabs.hidden = true;
    Object.values(authPanels).forEach((p) => { p.hidden = true; });
    document.getElementById("accountName").textContent = authUser.name;
    document.getElementById("accountRole").textContent = ROLE_LABELS[authUser.role] || authUser.role;
    panelAccount.hidden = false;
  } else {
    roleToggle.hidden = false;
    authTabs.hidden = false;
    panelAccount.hidden = true;
    setAuthTab("signin");
  }
}
function closeAuthModal() {
  authOverlay.hidden = true;
}

accountPill.addEventListener("click", openAuthModal);
accountPill.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openAuthModal(); }
});
authClose.addEventListener("click", closeAuthModal);
authOverlay.addEventListener("click", (e) => {
  if (e.target === authOverlay) closeAuthModal();
});

document.getElementById("quickChatBtn").addEventListener("click", () => {
  closeAuthModal();
  typedInput.focus();
});

const forgotPanel = document.getElementById("forgotPanel");
const forgotMessage = document.getElementById("forgotMessage");
document.getElementById("forgotPasswordLink").addEventListener("click", () => {
  forgotPanel.hidden = !forgotPanel.hidden;
});
document.getElementById("sendOtpBtn").addEventListener("click", async () => {
  const identifier = document.getElementById("forgotIdentifier").value.trim();
  if (!identifier) { forgotMessage.textContent = "Enter your email or phone first."; return; }
  try {
    const res = await fetch("/api/auth/forgot-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, channel: "email" }),
    });
    const data = await res.json();
    forgotMessage.textContent = data.message;
  } catch (err) {
    forgotMessage.textContent = `⚠️ ${err.message}`;
  }
});
document.getElementById("resetPasswordBtn").addEventListener("click", async () => {
  const identifier = document.getElementById("forgotIdentifier").value.trim();
  const otp = document.getElementById("resetOtp").value.trim();
  const new_password = document.getElementById("resetNewPassword").value;
  if (!identifier || !otp || !new_password) { forgotMessage.textContent = "Fill in email/phone, OTP, and new password."; return; }
  try {
    const res = await fetch("/api/auth/reset-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, otp, new_password }),
    });
    const data = await res.json();
    forgotMessage.textContent = data.message;
  } catch (err) {
    forgotMessage.textContent = `⚠️ ${err.message}`;
  }
});

function persistSession(user, token) {
  authUser = user;
  authToken = token;
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  updateAccountPill();
}

document.getElementById("registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    name: document.getElementById("registerName").value,
    email: document.getElementById("registerEmail").value,
    phone_number: "",
    password: document.getElementById("registerPassword").value,
    role: selectedRole,
    license_number: selectedRole === "lawyer" ? "" : null,
  };
  try {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    authMessage.textContent = data.message;
    if (data.success) {
      persistSession(data.user, data.token);
      closeAuthModal();
    }
  } catch (err) {
    authMessage.textContent = `⚠️ ${err.message}`;
  }
});

document.getElementById("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    identifier: document.getElementById("loginEmail").value,
    password: document.getElementById("loginPassword").value,
    role: selectedRole,
  };
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    authMessage.textContent = data.message;
    if (data.success) {
      persistSession(data.user, data.token);
      closeAuthModal();
    }
  } catch (err) {
    authMessage.textContent = `⚠️ ${err.message}`;
  }
});

document.getElementById("signOutBtn").addEventListener("click", async () => {
  try {
    if (authToken) {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: authToken }),
      });
    }
  } catch {
    // best-effort; clear the local session regardless of network failure
  }
  authUser = null;
  authToken = null;
  localStorage.removeItem(AUTH_USER_KEY);
  localStorage.removeItem(AUTH_TOKEN_KEY);
  updateAccountPill();
  closeAuthModal();
});

updateAccountPill();
setInterval(checkBackend, 15000);
