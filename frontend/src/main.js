// Wires the console UI to the real FastAPI backend (api.py). Vite's
// dev-server proxy (vite.config.js) forwards /api and /health to
// http://localhost:8000, so every call here is a same-origin relative path.

const transcript = document.getElementById("transcript");
const typedInput = document.getElementById("typedInput");
const btnSend = document.getElementById("btnSend");
const btnMic = document.getElementById("btnMic");
const btnUpload = document.getElementById("btnUpload");
const fileInput = document.getElementById("fileInput");
const btnLink = document.getElementById("btnLink");
const linkRow = document.getElementById("linkRow");
const videoLinkInput = document.getElementById("videoLinkInput");
const btnLinkSubmit = document.getElementById("btnLinkSubmit");
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

// ---------- Auth: sign in / sign up (individual or lawyer) -> /api/auth/* ----------

const authCard = document.getElementById("authCard");
const authSignedOut = document.getElementById("authSignedOut");
const authSignedIn = document.getElementById("authSignedIn");
const authRoleToggle = document.getElementById("authRoleToggle");
const authRoleButtons = authRoleToggle.querySelectorAll(".auth-role-btn");
const authRoleLabels = authCard.querySelectorAll(".authRoleLabel");
const authTabButtons = authCard.querySelectorAll(".auth-tab");
const signinForm = document.getElementById("signinForm");
const registerForm = document.getElementById("registerForm");
const registerLicense = document.getElementById("registerLicense");
const guestPanel = document.getElementById("guestPanel");
const authGuestBtn = document.getElementById("authGuestBtn");
const authGuestActive = document.getElementById("authGuestActive");
const authGuestLogoutBtn = document.getElementById("authGuestLogoutBtn");
const guestTabButton = authCard.querySelector('.auth-tab[data-tab="guest"]');
const forgotPasswordLink = document.getElementById("forgotPasswordLink");
const forgotPanel = document.getElementById("forgotPanel");
const forgotIdentifier = document.getElementById("forgotIdentifier");
const forgotSendBtn = document.getElementById("forgotSendBtn");
const forgotResetRow = document.getElementById("forgotResetRow");
const forgotOtp = document.getElementById("forgotOtp");
const forgotNewPassword = document.getElementById("forgotNewPassword");
const forgotResetBtn = document.getElementById("forgotResetBtn");
const forgotMessage = document.getElementById("forgotMessage");
const authMessage = document.getElementById("authMessage");
const authUserName = document.getElementById("authUserName");
const authUserRole = document.getElementById("authUserRole");
const authUserRoleIcon = document.getElementById("authUserRoleIcon");
const authSignOutBtn = document.getElementById("authSignOutBtn");
const consoleEl = document.querySelector(".console");

const ROLE_LABELS = { individual: "Individual", lawyer: "Lawyer" };
const ROLE_ICONS = { individual: "🙋", lawyer: "⚖️" };

// 8-12 characters, at least one lowercase, one uppercase, one digit, and one
// symbol — shared by the Sign Up password field and the forgot-password
// reset field so both enforce the same "strong" bar.
const STRONG_PASSWORD_RE = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,12}$/;
const PASSWORD_HINT = "Password must be 8-12 characters with upper & lower case, a number, and a symbol.";

let authRole = "individual";
let authTab = "signin";
let authUser = null;
let authToken = null;

function setAuthMessage(text, kind) {
  authMessage.textContent = text || "";
  authMessage.classList.toggle("error", kind === "error");
  authMessage.classList.toggle("success", kind === "success");
}

function setAuthRole(role) {
  authRole = role;
  authRoleButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.role === role));
  authRoleLabels.forEach((el) => (el.textContent = ROLE_LABELS[role]));
  registerLicense.hidden = role !== "lawyer";
  registerLicense.required = role === "lawyer";
  // Guest access is individual-only -- a lawyer account always needs to be
  // a real, verified account, so lawyers only get Sign In / Sign Up.
  guestTabButton.hidden = role === "lawyer";
  if (role === "lawyer" && authTab === "guest") setAuthTab("signin");
}

function setAuthTab(tab) {
  authTab = tab;
  authTabButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
  signinForm.hidden = tab !== "signin";
  registerForm.hidden = tab !== "register";
  guestPanel.hidden = tab !== "guest";
  // Forgot-password is reached from the Sign In tab only -- collapse it
  // (and any in-progress reset state) whenever a different tab is chosen.
  forgotPanel.hidden = true;
  forgotResetRow.hidden = true;
  setAuthMessage("");
  forgotMessage.textContent = "";
}

function revealConsole() {
  consoleEl.hidden = false;
}

authRoleButtons.forEach((btn) => btn.addEventListener("click", () => setAuthRole(btn.dataset.role)));
authTabButtons.forEach((btn) => btn.addEventListener("click", () => setAuthTab(btn.dataset.tab)));

// Loads this account's past chat history (see api.py's _account_session_id)
// into the transcript on sign-in/session-restore, so it's actually "cached"
// across reloads and devices instead of starting blank every time despite
// having a real account.
async function loadAccountChatHistory() {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/chat/${sessionId}`);
    const data = await res.json();
    for (const msg of data.messages || []) {
      addRow(msg.role, msg.role === "user" ? "You" : "AI", msg.text);
    }
  } catch {
    // Best-effort — an empty/missing history just means a fresh chat, same
    // as a brand-new account.
  }
}

function showSignedIn(user) {
  authUser = user;
  authSignedOut.hidden = true;
  authSignedIn.hidden = false;
  authUserName.textContent = user.name || user.email || "Signed in";
  authUserRole.textContent = user.role;
  authUserRoleIcon.textContent = ROLE_ICONS[user.role] || "👤";
  revealConsole();
  loadAccountChatHistory();
}

// Removes every transcript row except the static greeting bubble — used so
// a full log-out (see showSignedOut) doesn't leave a previous account's
// messages sitting on screen once the console is shown again for whoever
// signs in (or registers a guest session) next.
function resetTranscript() {
  transcript.querySelectorAll(".row").forEach((row) => {
    if (row.querySelector("#greetingBubble")) return;
    row.remove();
  });
}

function showSignedOut() {
  authUser = null;
  authToken = null;
  sessionId = null;
  localStorage.removeItem("legalAuthToken");
  localStorage.removeItem("legalAuthUser");
  authSignedOut.hidden = false;
  authSignedIn.hidden = true;
  // Full log-out, not just clearing the token: hide the chat console itself
  // (it only reappears via revealConsole on the next sign-in/guest) and wipe
  // any messages already on screen, so it isn't still showing the previous
  // account's conversation state behind the reopened auth card.
  consoleEl.hidden = true;
  resetTranscript();
}

async function restoreSession() {
  const token = localStorage.getItem("legalAuthToken");
  const storedUser = localStorage.getItem("legalAuthUser");
  if (!token || !storedUser) {
    if (sessionStorage.getItem("legalAuthGuest") === "1") {
      authSignedOut.hidden = true;
      authGuestActive.hidden = false;
      revealConsole();
    }
    return;
  }

  try {
    const res = await fetch("/api/auth/me", { headers: { Authorization: `Bearer ${token}` } });
    const data = await res.json();
    if (data.success) {
      authToken = token;
      sessionId = data.chat_session_id;
      showSignedIn(JSON.parse(storedUser));
    } else {
      showSignedOut();
    }
  } catch {
    // Backend unreachable at load time — keep the cached user so the UI
    // doesn't flash back to signed-out; checkBackend()'s poll will surface
    // the outage separately via the header pill. sessionId is left unset
    // here (not stored locally) — it'll be filled in once /api/auth/me
    // actually succeeds; until then a sent message would fall back to a
    // fresh random session rather than this account's history.
    authToken = token;
    showSignedIn(JSON.parse(storedUser));
  }
}

function persistSession(token, user, chatSessionId) {
  authToken = token;
  sessionId = chatSessionId;
  localStorage.setItem("legalAuthToken", token);
  localStorage.setItem("legalAuthUser", JSON.stringify(user));
  showSignedIn(user);
}

signinForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const identifier = document.getElementById("signinIdentifier").value.trim();
  const password = document.getElementById("signinPassword").value;
  setAuthMessage("Signing in…");

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, password, role: authRole }),
    });
    const data = await res.json();
    if (!data.success) {
      setAuthMessage(data.message || "Sign in failed.", "error");
      return;
    }
    persistSession(data.token, data.user, data.chat_session_id);
    setAuthMessage("");
    signinForm.reset();
  } catch (err) {
    setAuthMessage(`Couldn't reach the backend (${err.message}).`, "error");
  }
});

// ---------- Forgot password (Sign In tab, works for either role since
// lookup is by email/phone, not role) -> /api/auth/forgot-password + reset-password ----------

forgotPasswordLink.addEventListener("click", (e) => {
  e.preventDefault();
  forgotPanel.hidden = !forgotPanel.hidden;
  if (forgotPanel.hidden) {
    forgotResetRow.hidden = true;
    forgotMessage.textContent = "";
  } else {
    forgotIdentifier.value = document.getElementById("signinIdentifier").value.trim();
  }
});

forgotSendBtn.addEventListener("click", async () => {
  const identifier = forgotIdentifier.value.trim();
  if (!identifier) {
    forgotMessage.textContent = "Enter your email or phone first.";
    return;
  }
  forgotMessage.textContent = "Sending…";
  try {
    const res = await fetch("/api/auth/forgot-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, channel: "email" }),
    });
    const data = await res.json();
    // This demo backend has no real email/SMS sender -- it hands the OTP
    // straight back in the response, so surface it here too (data.otp)
    // instead of leaving the flow a dead end with nowhere the code
    // actually arrives.
    forgotMessage.textContent = data.otp ? `${data.message} — code: ${data.otp}` : data.message;
    if (data.success) forgotResetRow.hidden = false;
  } catch (err) {
    forgotMessage.textContent = `Couldn't reach the backend (${err.message}).`;
  }
});

forgotResetBtn.addEventListener("click", async () => {
  const identifier = forgotIdentifier.value.trim();
  const otp = forgotOtp.value.trim();
  const newPassword = forgotNewPassword.value;
  if (!identifier || !otp || !newPassword) {
    forgotMessage.textContent = "Fill in the email/phone, code, and new password.";
    return;
  }
  if (!STRONG_PASSWORD_RE.test(newPassword)) {
    forgotMessage.textContent = PASSWORD_HINT;
    return;
  }
  forgotMessage.textContent = "Resetting…";
  try {
    const res = await fetch("/api/auth/reset-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, otp, new_password: newPassword }),
    });
    const data = await res.json();
    forgotMessage.textContent = data.message;
    if (data.success) {
      forgotOtp.value = "";
      forgotNewPassword.value = "";
      forgotResetRow.hidden = true;
    }
  } catch (err) {
    forgotMessage.textContent = `Couldn't reach the backend (${err.message}).`;
  }
});

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    name: document.getElementById("registerName").value.trim(),
    email: document.getElementById("registerEmail").value.trim(),
    phone_number: document.getElementById("registerPhone").value.trim(),
    password: document.getElementById("registerPassword").value,
    role: authRole,
    license_number: authRole === "lawyer" ? registerLicense.value.trim() : null,
  };
  if (!STRONG_PASSWORD_RE.test(payload.password)) {
    setAuthMessage(PASSWORD_HINT, "error");
    return;
  }
  setAuthMessage("Creating your account…");

  try {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.success) {
      setAuthMessage(data.message || "Sign up failed.", "error");
      return;
    }
    persistSession(data.token, data.user, data.chat_session_id);
    setAuthMessage("");
    registerForm.reset();
  } catch (err) {
    setAuthMessage(`Couldn't reach the backend (${err.message}).`, "error");
  }
});

function enterGuestMode() {
  authSignedOut.hidden = true;
  authGuestActive.hidden = false;
  sessionStorage.setItem("legalAuthGuest", "1");
  revealConsole();
}

authGuestBtn.addEventListener("click", enterGuestMode);

authGuestLogoutBtn.addEventListener("click", () => {
  sessionStorage.removeItem("legalAuthGuest");
  authGuestActive.hidden = true;
  authSignedOut.hidden = false;
  consoleEl.hidden = true;
  resetTranscript();
});

authSignOutBtn.addEventListener("click", async () => {
  if (authToken) {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: authToken }),
      });
    } catch {
      // Best-effort — still clear the local session below even if this fails.
    }
  }
  showSignedOut();
});

const authDeleteBtn = document.getElementById("authDeleteBtn");
authDeleteBtn.addEventListener("click", async () => {
  if (!authToken) return;
  const confirmed = window.confirm(
    "Delete your account? This permanently removes your account and chat history and can't be undone."
  );
  if (!confirmed) return;

  try {
    const res = await fetch("/api/auth/account", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${authToken}` },
    });
    const data = await res.json();
    if (!data.success) {
      window.alert(data.message || "Couldn't delete the account.");
      return;
    }
  } catch (err) {
    window.alert(`Couldn't reach the backend (${err.message}).`);
    return;
  }
  showSignedOut();
});

setAuthRole(authRole);
setAuthTab(authTab);
restoreSession();

// ---------- Text-to-speech: read AI replies aloud via the browser's free,
// built-in Web Speech API (window.speechSynthesis) -- no backend call, no
// API key, no per-request cost. Auto-speaks only when the question came in
// by voice (see sendMessage's viaVoice option) — a typed question gets a
// typed-feeling reply, a spoken question gets a spoken-back one; every bot
// reply also keeps its own 🔊 replay button (see addRow) for manual re-play
// regardless of how the question was asked. Voice quality/availability
// (including whether an Armenian voice exists at all) depends entirely on
// the user's OS/browser; there's no server-side fallback. ----------

const TTS_SUPPORTED = "speechSynthesis" in window;
const TTS_LANG_PREFIX = { hy: "hy", en: "en", ru: "ru" };
let ttsVoices = [];

function refreshVoices() {
  ttsVoices = window.speechSynthesis.getVoices();
}
if (TTS_SUPPORTED) {
  refreshVoices();
  window.speechSynthesis.addEventListener("voiceschanged", refreshVoices);
}

function pickVoice(lang) {
  const prefix = TTS_LANG_PREFIX[lang] || lang;
  return (
    ttsVoices.find((v) => v.lang.toLowerCase().startsWith(prefix)) ||
    ttsVoices.find((v) => v.lang.toLowerCase().startsWith("en")) ||
    ttsVoices[0]
  );
}

function speakText(text, lang) {
  if (!TTS_SUPPORTED || !text) return;
  window.speechSynthesis.cancel(); // one reply at a time, no overlapping queue
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(lang);
  if (voice) utterance.voice = voice;
  utterance.lang = voice ? voice.lang : (TTS_LANG_PREFIX[lang] || "en");
  window.speechSynthesis.speak(utterance);
}

function scrollToEnd() {
  transcript.scrollTop = transcript.scrollHeight;
}

function addRow(kind, who, content, opts = {}) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;

  // Plain-text bot replies (not the typing indicator or HTML-rendered video
  // analysis) get a replay button so any past reply can be re-heard on
  // demand, independent of the "read replies aloud" auto-speak toggle.
  const speakable = kind === "bot" && who && !opts.html;

  if (who) {
    const whoRow = document.createElement("div");
    whoRow.className = "who-row";
    const whoEl = document.createElement("span");
    whoEl.className = "who";
    whoEl.textContent = who;
    whoRow.appendChild(whoEl);
    if (speakable) {
      const speakBtn = document.createElement("button");
      speakBtn.type = "button";
      speakBtn.className = "speak-btn";
      speakBtn.title = "Read this reply aloud";
      speakBtn.textContent = "🔊";
      speakBtn.addEventListener("click", () => speakText(content, currentLanguage));
      whoRow.appendChild(speakBtn);
    }
    row.appendChild(whoRow);
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
  btnLink.disabled = !on;
  btnLinkSubmit.disabled = !on;
  videoLinkInput.disabled = !on;
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

async function sendMessage(message, { viaVoice = false } = {}) {
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
      const replyText = data.response || "(empty response)";
      addRow("bot", "AI", replyText);
      if (viaVoice) speakText(replyText, currentLanguage);
    }
    setBackendStatus(true);
  } catch (err) {
    typing.row.remove();
    addRow(
      "system", null,
      `⚠️ Couldn't reach the backend (${err.message}). Is "uvicorn api:app --reload --port 8010" running?`,
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
    sendMessage(data.text, { viaVoice: true });
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

// ---------- Upload: doc -> case DB, video (file or link) -> action/emotion + speech transcript ----------

// Vision/Whisper models now warm up at backend startup (see api.py's
// warm_up_legal_agent), not on the first request, but a slow yt-dlp
// download or a genuinely stuck/dropped backend connection can still hang
// the fetch indefinitely with no timeout — the browser's own network stack
// eventually gives up and throws its own opaque error (e.g. Safari's bare
// "Load failed", with no indication of what happened or how long it waited).
// Aborting ourselves first means the UI always shows a clear, actionable
// message instead.
const VIDEO_FETCH_TIMEOUT_MS = 4 * 60 * 1000;

async function fetchWithTimeout(url, options = {}, timeoutMs = VIDEO_FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`No response after ${Math.round(timeoutMs / 1000)}s — the backend may be stuck, or this video is taking unusually long.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Shared by the file-upload flow and the video-link flow below — both POST
// to a backend endpoint that returns the same {kind: "video", ...} shape
// (see api.py's _analyze_video_upload), so there's one place that turns
// that response into a chat bubble instead of two drifting copies.
function renderVideoAnalysisRow(data) {
  const actions = data.actions && data.actions.length ? data.actions.join(", ") : "none detected";
  let spoken;
  if (data.transcript) {
    spoken = `<br><strong>🗣️ What was said:</strong> "${data.transcript}"`;
  } else if (data.has_nonspeech_audio) {
    spoken = "<br><strong>🎵 Audio:</strong> sound was heard (possibly music or background audio), but no speech was recognized";
  } else {
    spoken = `<br><strong>🗣️ What was said:</strong> no speech detected — based on action detection, here's what's happening: ${actions}`;
  }
  const emotionChanges = data.emotion_changes && data.emotion_changes.length > 1
    ? `<br><strong>Emotion changed during video:</strong> ${data.emotion_changes.join(" → ")}`
    : "";
  const objects = data.objects && data.objects.length
    ? `<br><strong>Objects seen:</strong> ${data.objects.join(", ")}`
    : "";
  addRow(
    "bot", "AI",
    `✅ Analyzed ${data.frames_analyzed} sampled frame(s).<br>` +
    `<strong>Actions:</strong> ${actions}<br>` +
    `<strong>Emotion:</strong> ${data.emotion || "n/a"}` +
    emotionChanges + objects +
    spoken,
    { html: true }
  );
}

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
      ? `🎥 Analyzing ${f.name}… (action/emotion detection + speech transcript)`
      : `📄 Processing ${f.name}…`
  );

  try {
    const form = new FormData();
    form.append("file", f);
    form.append("language", currentLanguage);
    if (sessionId) form.append("session_id", sessionId);

    const res = await fetchWithTimeout("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    working.row.remove();

    if (!res.ok || data.success === false) {
      addRow("system", null, data.message || `HTTP ${res.status}`, { warn: true });
    } else if (data.kind === "video") {
      renderVideoAnalysisRow(data);
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

// ---------- Video link (YouTube / TikTok / Instagram / etc. via /api/upload-link) ----------

btnLink.addEventListener("click", () => {
  if (busy) return;
  const showing = !linkRow.hidden;
  linkRow.hidden = showing;
  if (!showing) videoLinkInput.focus();
});

async function submitVideoLink() {
  const url = videoLinkInput.value.trim();
  if (!url || busy) return;

  setComposerEnabled(false);
  const working = addRow(
    "system", null,
    `🔗 Downloading and analyzing ${url}… (can take a while depending on the video's length)`
  );

  try {
    const form = new FormData();
    form.append("url", url);
    form.append("language", currentLanguage);
    if (sessionId) form.append("session_id", sessionId);

    const res = await fetchWithTimeout("/api/upload-link", { method: "POST", body: form });
    const data = await res.json();
    working.row.remove();

    if (!res.ok || data.success === false) {
      addRow("system", null, data.message || `HTTP ${res.status}`, { warn: true });
    } else {
      renderVideoAnalysisRow(data);
      videoLinkInput.value = "";
      linkRow.hidden = true;
    }
  } catch (err) {
    working.row.remove();
    addRow("system", null, `⚠️ Video link analysis failed: ${err.message}`, { warn: true });
  } finally {
    setComposerEnabled(true);
  }
}

btnLinkSubmit.addEventListener("click", submitVideoLink);
videoLinkInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitVideoLink();
});

document.addEventListener("keydown", (e) => {
  if (document.activeElement === typedInput || document.activeElement === videoLinkInput) return;
  if (e.key === "t") typedInput.focus();
  if (e.key === "m") btnMic.click();
  if (e.key === "u") btnUpload.click();
  if (e.key === "l") btnLink.click();
});
