// Client-side LLM chat demo. Five providers, all running fully in-browser:
//   - "webllm": open-weight tiny chat models via WebGPU (@mlc-ai/web-llm)
//   - "transformers": open-weight tiny chat models via WASM/WebGPU (@huggingface/transformers)
//   - "caption": one-shot image captioning via WASM/WebGPU (@huggingface/transformers)
//   - "vqa": multi-turn "chat about an image" via WASM/WebGPU (@huggingface/transformers)
//   - "chrome": Chrome's built-in Gemini Nano (window.LanguageModel Prompt API)

// sizeEstimate is a rough, approximate download size shown in the confirm
// dialog before committing to a download — not measured at runtime.
const providerModels = {
  webllm: [
    { value: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B Instruct (q4f16)", sizeEstimate: "~880 MB" },
    { value: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 0.5B Instruct (q4f16)", sizeEstimate: "~450 MB" },
  ],
  transformers: [
    { value: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen 2.5 0.5B Instruct (ONNX, q4)", dtype: "q4", sizeEstimate: "~300 MB" },
    { value: "HuggingFaceTB/SmolLM2-135M-Instruct", label: "SmolLM2 135M Instruct (ONNX, very small/fast)", sizeEstimate: "~130 MB" },
  ],
  caption: [
    // The default quantized export of this model has a broken decoder graph
    // ("Missing required scale" DequantizeLinear error) under current ONNX
    // Runtime Web — fp32 sidesteps it at the cost of a bigger download.
    { value: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Image Captioning", dtype: "fp32", sizeEstimate: "~800 MB" },
  ],
  vqa: [
    { value: "HuggingFaceTB/SmolVLM-500M-Instruct", label: "SmolVLM 500M Instruct (image Q&A)", dtype: "q4", sizeEstimate: "~340 MB" },
  ],
  chrome: [],
};

const providerSelect = document.getElementById("provider");
const modelSelect = document.getElementById("model");
const modelGroup = document.getElementById("model-group");
const loadBtn = document.getElementById("load-btn");
const statusEl = document.getElementById("status");
const chatEl = document.getElementById("chat");
const chatInner = document.getElementById("chat-inner");
const form = document.getElementById("chat-form");
const textControls = document.getElementById("text-controls");
const imageControls = document.getElementById("image-controls");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const imageInput = document.getElementById("image-input");
const imageLabel = document.getElementById("image-label");
const vqaImageInput = document.getElementById("vqa-image-input");
const vqaAttachLabel = document.getElementById("vqa-attach-label");
const loadModal = document.getElementById("load-modal");
const modalTitle = document.getElementById("modal-title");
const modalModelName = document.getElementById("modal-model-name");
const modalConfirmSection = document.getElementById("modal-confirm-section");
const modalSizeWarning = document.getElementById("modal-size-warning");
const modalCancelBtn = document.getElementById("modal-cancel-btn");
const modalConfirmBtn = document.getElementById("modal-confirm-btn");
const modalProgressSection = document.getElementById("modal-progress-section");
const modalProgressFill = document.getElementById("modal-progress-fill");
const modalStatus = document.getElementById("modal-status");
const modalCloseBtn = document.getElementById("modal-close-btn");
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const settingsStorageSummary = document.getElementById("settings-storage-summary");
const settingsList = document.getElementById("settings-list");
const settingsCloseBtn = document.getElementById("settings-close-btn");
const settingsClearAllBtn = document.getElementById("settings-clear-all-btn");
const topbarSettingsBtn = document.getElementById("topbar-settings-btn");
const menuModal = document.getElementById("menu-modal");
const menuStorageBtn = document.getElementById("menu-storage-btn");
const menuAboutBtn = document.getElementById("menu-about-btn");
const menuCloseBtn = document.getElementById("menu-close-btn");
const aboutModal = document.getElementById("about-modal");
const aboutCloseBtn = document.getElementById("about-close-btn");
const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const sidebarToggle = document.getElementById("sidebar-toggle");
const newChatBtn = document.getElementById("new-chat-btn");
const topbarTitle = document.getElementById("topbar-title");
const mainEl = document.querySelector(".main");
const composerHeroTitle = document.getElementById("composer-hero-title");
const composerHeroLoadBtn = document.getElementById("composer-hero-load-btn");
const sidebarChatsEl = document.getElementById("sidebar-chats");
const clearHistoryBtn = document.getElementById("clear-history-btn");

/** @type {{role: "user"|"assistant", content: string}[]} */
let history = [];

// Active provider handle. Exactly one of these is set after a successful load.
let webllmEngine = null;
// transformersPipeline/captionerPipeline are just booleans here — the real
// pipeline objects live inside transformersWorker (see below), never on the
// main thread, so a long generation can never block this tab's UI.
let transformersPipeline = null;
let captionerPipeline = null;
let vqaPipeline = null;
// "provider:modelValue" of whichever engine is actually loaded right now —
// distinct from providerSelect/modelSelect, which just reflect the current
// dropdown selection and can change without a new load happening.
let loadedModelKey = null;
let chromeSession = null;

// VQA ("chat about an image") state — deliberately separate from `history`:
// it's never persisted to `chatSessions`/localStorage (see clearChat()), and
// its first turn embeds an image, a shape the text-chat providers never use.
let vqaImage = null;
/** @type {{role: "user"|"assistant", content: string}[]} */
let vqaMessages = [];

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// Downloading a model is a real commitment — tens to hundreds of MB over the
// network — so require an explicit yes/no confirmation naming the size before
// anything downloads, then make progress obvious with a blocking full-page
// modal and progress bar rather than a quiet status line.
function isModalCancelable() {
  return (
    !modalConfirmSection.classList.contains("hidden") ||
    !modalCloseBtn.classList.contains("hidden")
  );
}

function hideLoadModal() {
  loadModal.classList.add("hidden");
}

function dismissModalIfCancelable() {
  if (isModalCancelable()) hideLoadModal();
}

function showConfirmModal() {
  const config = getSelectedModelConfig();
  const modelLabel =
    config?.label ?? providerSelect.options[providerSelect.selectedIndex].textContent;

  modalTitle.textContent = "Download model?";
  modalModelName.textContent = modelLabel;
  modalSizeWarning.textContent =
    providerSelect.value === "chrome"
      ? "Chrome manages this download itself — size and progress aren't reported to this page."
      : `Estimated download size: ${config?.sizeEstimate ?? "unknown"} (approximate).`;

  modalConfirmSection.classList.remove("hidden");
  modalProgressSection.classList.add("hidden");
  loadModal.classList.remove("hidden");
}

function setLoadProgress(text, percent = null) {
  setStatus(text);
  modalStatus.textContent = text;
  if (percent === null) {
    modalProgressFill.classList.add("indeterminate");
  } else {
    modalProgressFill.classList.remove("indeterminate");
    modalProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
}

modalCancelBtn.addEventListener("click", hideLoadModal);
modalCloseBtn.addEventListener("click", hideLoadModal);
loadModal.addEventListener("click", (e) => {
  if (e.target === loadModal) dismissModalIfCancelable();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !loadModal.classList.contains("hidden")) {
    dismissModalIfCancelable();
  }
});

// Storage inspector: models are cached via the Cache API (see providerModels
// above for which model each URL belongs to), not localStorage/sessionStorage
// — so this reads directly from `caches` rather than tracking state ourselves.
function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

const providerLabels = {
  webllm: "WebLLM",
  transformers: "Transformers.js",
  caption: "Transformers.js (captioning)",
  vqa: "Transformers.js (image Q&A)",
  chrome: "Chrome built-in AI",
};

function allCacheableModels() {
  return Object.entries(providerModels)
    .filter(([provider]) => provider !== "chrome")
    .flatMap(([provider, models]) => models.map((m) => ({ provider, ...m })));
}

// Model URLs on Hugging Face always contain the model's repo id, so matching
// by substring works regardless of which cache name the library chose or
// what org prefix it added — no need to hardcode cache names.
async function findCachedEntriesFor(modelValue) {
  const cacheNames = await caches.keys();
  const matches = [];
  for (const name of cacheNames) {
    const cache = await caches.open(name);
    for (const request of await cache.keys()) {
      if (request.url.includes(modelValue)) {
        matches.push({ cacheName: name, request });
      }
    }
  }
  return matches;
}

async function sizeOfEntries(entries) {
  let total = 0;
  for (const { cacheName, request } of entries) {
    const cache = await caches.open(cacheName);
    const response = await cache.match(request);
    if (!response) continue;
    const contentLength = response.headers.get("content-length");
    total += contentLength
      ? Number(contentLength)
      : (await response.clone().blob()).size;
  }
  return total;
}

async function deleteCachedEntriesFor(modelValue) {
  for (const { cacheName, request } of await findCachedEntriesFor(modelValue)) {
    const cache = await caches.open(cacheName);
    await cache.delete(request);
  }
}

async function renderStorageSummary() {
  if (!navigator.storage?.estimate) {
    settingsStorageSummary.textContent = "";
    return;
  }
  const { usage, quota } = await navigator.storage.estimate();
  settingsStorageSummary.textContent = quota
    ? `Using ${formatBytes(usage)} of roughly ${formatBytes(quota)} available to this site.`
    : `Using ${formatBytes(usage)}.`;
}

async function renderSettingsList() {
  if (!window.caches) {
    settingsList.innerHTML =
      '<p class="modal-note">This browser does not support the Cache API, so downloaded models can\'t be inspected here.</p>';
    return;
  }

  settingsList.innerHTML = '<p class="modal-note">Scanning browser storage…</p>';

  const rows = await Promise.all(
    allCacheableModels().map(async (m) => {
      const entries = await findCachedEntriesFor(m.value);
      const size = entries.length ? await sizeOfEntries(entries) : 0;
      return { ...m, downloaded: entries.length > 0, size };
    })
  );

  settingsList.innerHTML = "";

  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "settings-row";

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "settings-row-name";
    name.textContent = row.label;
    const meta = document.createElement("div");
    meta.className = "settings-row-meta";
    meta.textContent = row.downloaded
      ? `${providerLabels[row.provider]} · Downloaded · ${formatBytes(row.size)}`
      : `${providerLabels[row.provider]} · Not downloaded`;
    info.append(name, meta);
    el.appendChild(info);

    if (row.downloaded) {
      const clearBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.className = "btn btn-secondary settings-row-clear";
      clearBtn.textContent = "Clear";
      clearBtn.addEventListener("click", async () => {
        clearBtn.disabled = true;
        clearBtn.textContent = "Clearing…";
        await deleteCachedEntriesFor(row.value);
        await Promise.all([renderSettingsList(), renderStorageSummary()]);
      });
      el.appendChild(clearBtn);
    }

    settingsList.appendChild(el);
  }

  const chromeRow = document.createElement("div");
  chromeRow.className = "settings-row";
  chromeRow.innerHTML =
    '<div><div class="settings-row-name">Gemini Nano</div>' +
    '<div class="settings-row-meta">Chrome built-in · managed by Chrome, not visible to this page</div></div>';
  settingsList.appendChild(chromeRow);
}

function openSettingsModal() {
  settingsModal.classList.remove("hidden");
  renderSettingsList();
  renderStorageSummary();
}

function closeSettingsModal() {
  settingsModal.classList.add("hidden");
}

settingsBtn.addEventListener("click", openSettingsModal);
settingsCloseBtn.addEventListener("click", closeSettingsModal);
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettingsModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !settingsModal.classList.contains("hidden")) {
    closeSettingsModal();
  }
});

settingsClearAllBtn.addEventListener("click", async () => {
  settingsClearAllBtn.disabled = true;
  settingsClearAllBtn.textContent = "Clearing…";
  for (const m of allCacheableModels()) {
    await deleteCachedEntriesFor(m.value);
  }
  settingsClearAllBtn.disabled = false;
  settingsClearAllBtn.textContent = "Clear all downloaded models";
  await Promise.all([renderSettingsList(), renderStorageSummary()]);
});

// The gear icon opens a small "links" modal (Storage / About) rather than a
// dropdown, matching the same modal-overlay pattern used everywhere else.
function openMenuModal() {
  menuModal.classList.remove("hidden");
}

function closeMenuModal() {
  menuModal.classList.add("hidden");
}

function openAboutModal() {
  aboutModal.classList.remove("hidden");
}

function closeAboutModal() {
  aboutModal.classList.add("hidden");
}

topbarSettingsBtn.addEventListener("click", openMenuModal);
menuCloseBtn.addEventListener("click", closeMenuModal);
menuModal.addEventListener("click", (e) => {
  if (e.target === menuModal) closeMenuModal();
});

menuStorageBtn.addEventListener("click", () => {
  closeMenuModal();
  openSettingsModal();
});
menuAboutBtn.addEventListener("click", () => {
  closeMenuModal();
  openAboutModal();
});

aboutCloseBtn.addEventListener("click", closeAboutModal);
aboutModal.addEventListener("click", (e) => {
  if (e.target === aboutModal) closeAboutModal();
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!aboutModal.classList.contains("hidden")) closeAboutModal();
  else if (!menuModal.classList.contains("hidden")) closeMenuModal();
});

function setChatEnabled(enabled) {
  // In VQA mode, the text input additionally requires an image to already be
  // attached for the current session — nothing to ask a question about yet.
  const vqaReady = providerSelect.value !== "vqa" || vqaImage !== null;
  input.disabled = !enabled || !vqaReady;
  sendBtn.disabled = !enabled || !vqaReady;
  imageInput.disabled = !enabled;
  imageLabel.classList.toggle("disabled", !enabled);
  vqaImageInput.disabled = !enabled;
  // The empty-state hero shows the *selected* model's name regardless of
  // whether it's actually loaded, so a greyed-out, unexplained input is the
  // only other cue — easy to miss, especially on mobile where the sidebar's
  // "Load model" button starts off-screen. Surface the same action here,
  // recomputed from loadedModelKey (not the `enabled` param) so it stays
  // correct even in the VQA case where enabled=true but vqaReady=false.
  const modelIsLoaded =
    loadedModelKey === `${providerSelect.value}:${modelSelect.value}`;
  composerHeroLoadBtn.classList.toggle("hidden", modelIsLoaded);
}

// Before any messages exist, the composer floats centered in the pane
// (see .main.is-empty in style.css); once there's content, it's pinned to
// the bottom like a normal chat.
function updateEmptyState() {
  mainEl.classList.toggle("is-empty", chatInner.children.length === 0);
}

function addBubble(role) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;

  row.append(avatar, bubble);
  chatInner.appendChild(row);
  updateEmptyState();
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function addImageBubble(file) {
  const el = addBubble("user");
  const img = document.createElement("img");
  img.className = "thumb";
  img.alt = "uploaded image";
  img.src = URL.createObjectURL(file);
  img.onload = () => URL.revokeObjectURL(img.src);
  el.appendChild(img);
  return el;
}

function populateModelOptions(provider) {
  const models = providerModels[provider];
  modelSelect.innerHTML = "";
  for (const { value, label } of models) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    modelSelect.appendChild(opt);
  }
  modelGroup.style.display = models.length ? "" : "none";
}

function getSelectedModelConfig() {
  return providerModels[providerSelect.value].find(
    (m) => m.value === modelSelect.value
  );
}

function syncInputMode(provider) {
  const isCaption = provider === "caption";
  textControls.classList.toggle("hidden", isCaption);
  imageControls.classList.toggle("hidden", !isCaption);
  vqaAttachLabel.classList.toggle("hidden", provider !== "vqa");
  input.placeholder =
    provider === "vqa" ? "Attach an image to start…" : "Say something...";
}

// Clears the visible conversation only — the loaded engine (if any) stays
// loaded, so "New chat" behaves like ChatGPT's: same model, blank history.
// Also drops the current *saved* session pointer: the next successful reply
// starts a fresh entry in the sidebar rather than overwriting the old one.
function clearChat() {
  history = [];
  chatInner.innerHTML = "";
  imageInput.value = "";
  vqaImageInput.value = "";
  vqaImage = null;
  vqaMessages = [];
  if (providerSelect.value === "vqa") input.placeholder = "Attach an image to start…";
  currentSessionId = null;
  updateEmptyState();
  // Re-derive enabled state now that vqaImage was just reset — without this,
  // an input left enabled from a prior exchange stays enabled with no image
  // attached, letting a text-only submit through with a null image.
  setChatEnabled(loadedModelKey === `${providerSelect.value}:${modelSelect.value}`);
}

function resetEngineHandles() {
  webllmEngine = null;
  transformersPipeline = null;
  captionerPipeline = null;
  vqaPipeline = null;
  chromeSession = null;
  loadedModelKey = null;
  terminateTransformersWorker();
}

// Used when switching provider/model via the dropdowns: the old engine and
// the displayed conversation are both invalid, so both get cleared.
function resetProviderState() {
  resetEngineHandles();
  clearChat();
  setChatEnabled(false);
}

function updateTopbarTitle() {
  const config = getSelectedModelConfig();
  const modelLabel =
    config?.label ?? providerSelect.options[providerSelect.selectedIndex]?.textContent;
  const title = modelLabel
    ? `${providerLabels[providerSelect.value] ?? providerSelect.value} · ${modelLabel}`
    : "No model loaded";
  topbarTitle.textContent = title;
  composerHeroTitle.textContent = title;
}

// Chat history: saved sessions are plain text (a few KB each at most), so
// unlike model weights, localStorage is the right tool here — no need for
// the Cache API. Only text-chat providers (webllm/transformers/chrome) get
// saved; image captioning is a one-shot action rather than a conversation,
// so it deliberately isn't part of `history` and isn't persisted.
let chatSessions = [];
try {
  chatSessions = JSON.parse(localStorage.getItem("chatSessions") || "[]");
} catch (err) {
  console.error("Failed to read saved chat history", err);
}
let currentSessionId = null;

function saveSessions() {
  try {
    localStorage.setItem("chatSessions", JSON.stringify(chatSessions));
  } catch (err) {
    console.error("Failed to save chat history", err);
  }
}

function deriveTitle(text) {
  const trimmed = text.trim();
  if (!trimmed) return "New conversation";
  return trimmed.length > 42 ? `${trimmed.slice(0, 42)}…` : trimmed;
}

// Called after every exchange (success or failure) so the sent message is
// never silently lost, even if the model failed to reply.
function upsertCurrentSession() {
  if (history.length === 0) return;

  const now = Date.now();
  const existing = chatSessions.find((s) => s.id === currentSessionId);

  if (existing) {
    existing.history = history.map((m) => ({ ...m }));
    existing.updatedAt = now;
  } else {
    const firstUserMessage = history.find((m) => m.role === "user");
    const session = {
      id: `chat_${now}_${Math.random().toString(36).slice(2, 8)}`,
      title: deriveTitle(firstUserMessage?.content ?? ""),
      provider: providerSelect.value,
      modelValue: modelSelect.value,
      history: history.map((m) => ({ ...m })),
      createdAt: now,
      updatedAt: now,
    };
    chatSessions.push(session);
    currentSessionId = session.id;
  }

  saveSessions();
  renderSidebarChats();
}

function renderSidebarChats() {
  sidebarChatsEl.innerHTML = "";

  if (chatSessions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "sidebar-chat-empty";
    empty.textContent = "No saved chats yet";
    sidebarChatsEl.appendChild(empty);
    return;
  }

  const sorted = [...chatSessions].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const session of sorted) {
    const item = document.createElement("div");
    item.className = `sidebar-chat-item${session.id === currentSessionId ? " active" : ""}`;
    item.addEventListener("click", () => loadSession(session.id));

    const title = document.createElement("span");
    title.className = "sidebar-chat-title";
    title.textContent = session.title;
    item.appendChild(title);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "sidebar-chat-delete";
    deleteBtn.setAttribute("aria-label", `Delete "${session.title}"`);
    deleteBtn.innerHTML =
      '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>';
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session.id);
    });
    item.appendChild(deleteBtn);

    sidebarChatsEl.appendChild(item);
  }
}

function loadSession(id) {
  const session = chatSessions.find((s) => s.id === id);
  if (!session) return;

  currentSessionId = id;
  history = session.history.map((m) => ({ ...m }));

  providerSelect.value = session.provider;
  populateModelOptions(session.provider);
  syncInputMode(session.provider);
  modelSelect.value = session.modelValue;
  updateTopbarTitle();

  chatInner.innerHTML = "";
  for (const msg of history) {
    const bubble = addBubble(msg.role);
    if (msg.role === "assistant") {
      renderAssistantMarkdown(bubble, msg.content);
    } else {
      bubble.textContent = msg.content;
    }
  }

  const matchesLoaded = loadedModelKey === `${session.provider}:${session.modelValue}`;
  setChatEnabled(matchesLoaded);
  setStatus(matchesLoaded ? "" : "Load this model to continue the conversation.");

  renderSidebarChats();
  if (isMobileViewport()) setSidebarOpen(false);
}

function deleteSession(id) {
  chatSessions = chatSessions.filter((s) => s.id !== id);
  saveSessions();
  if (id === currentSessionId) clearChat();
  renderSidebarChats();
}

function clearAllSessions() {
  chatSessions = [];
  saveSessions();
  clearChat();
  renderSidebarChats();
}

clearHistoryBtn.addEventListener("click", clearAllSessions);

providerSelect.addEventListener("change", () => {
  populateModelOptions(providerSelect.value);
  syncInputMode(providerSelect.value);
  resetProviderState();
  setStatus("");
  updateTopbarTitle();
});

modelSelect.addEventListener("change", updateTopbarTitle);

populateModelOptions(providerSelect.value);
syncInputMode(providerSelect.value);
updateTopbarTitle();
renderSidebarChats();
updateEmptyState();

function isMobileViewport() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function setSidebarOpen(open) {
  sidebar.classList.toggle("collapsed", !open);
  sidebarBackdrop.classList.toggle("hidden", !(open && isMobileViewport()));
  localStorage.setItem("sidebarOpen", String(open));
}

function toggleSidebar() {
  setSidebarOpen(sidebar.classList.contains("collapsed"));
}

const savedSidebarOpen = localStorage.getItem("sidebarOpen");
setSidebarOpen(savedSidebarOpen !== null ? savedSidebarOpen === "true" : !isMobileViewport());

sidebarToggle.addEventListener("click", toggleSidebar);
sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));

// Crossing the 768px breakpoint after load (e.g. resizing the window, or
// DevTools' device toolbar without a hard reload) leaves the CSS-driven
// sidebar overlay and the JS-driven backdrop out of sync unless resynced.
window.addEventListener("resize", () => setSidebarOpen(!sidebar.classList.contains("collapsed")));

function handleNewChat() {
  clearChat();
  if (isMobileViewport()) setSidebarOpen(false);
}

newChatBtn.addEventListener("click", handleNewChat);

// Hugging Face's CDN occasionally serves a transient error page instead of
// the real model file; browsers report that as a CORS failure since error
// responses don't carry CORS headers. Both WebLLM and Transformers.js pull
// model weights from Hugging Face, so both benefit from a short retry.
async function withRetries(fn, { maxAttempts = 3 } = {}) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === maxAttempts) {
        throw new Error(
          `${err.message} (failed after ${maxAttempts} attempts — this is ` +
            "usually a transient Hugging Face CDN hiccup, try Load model again)"
        );
      }
      setLoadProgress(`Download hiccup, retrying (${attempt}/${maxAttempts})...`);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
}

async function loadWebLLM() {
  setLoadProgress("Loading WebLLM engine...");
  const { CreateMLCEngine } = await import(
    "https://esm.run/@mlc-ai/web-llm"
  );

  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this browser. Try a recent Chrome/Edge."
    );
  }

  const modelId = modelSelect.value;
  webllmEngine = await withRetries(() =>
    CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        setLoadProgress(
          report.text || "Loading model...",
          report.progress != null ? report.progress * 100 : null
        );
      },
    })
  );
  setLoadProgress(`Loaded ${modelId}. Ready to chat.`, 100);
}

function transformersProgressCallback(progress) {
  if (progress.status === "progress") {
    const pct = Math.round(progress.progress ?? 0);
    setLoadProgress(`Downloading ${progress.file} (${pct}%)...`, pct);
  } else {
    setLoadProgress(`${progress.status}...`);
  }
}

// Transformers.js runs entirely inside this worker (model loading, chat
// generation, and image captioning) rather than on the main thread. WASM
// inference is synchronous, so running it here means a long reply can never
// freeze the tab's UI or trigger Chrome's "Page Unresponsive" prompt.
let transformersWorker = null;

// The worker script is only ever fetched from inside a click handler (Load
// model), never during the page's initial navigation — so a hard-reload of
// index.html does NOT force-bypass the browser's cache for it, and the
// plain python3 -m http.server dev setup sends no cache-control headers to
// stop the browser reusing a stale copy indefinitely. Bump this by hand
// whenever transformers-worker.js changes so the URL actually changes and
// forces a real re-fetch.
const WORKER_VERSION = "2";

function getTransformersWorker() {
  if (!transformersWorker) {
    transformersWorker = new Worker(`transformers-worker.js?v=${WORKER_VERSION}`, { type: "module" });
  }
  return transformersWorker;
}

function terminateTransformersWorker() {
  if (transformersWorker) {
    transformersWorker.terminate();
    transformersWorker = null;
  }
}

function workerLoad(task, modelId, dtype, onProgress, messageType = "load", device = undefined) {
  const worker = getTransformersWorker();
  return new Promise((resolve, reject) => {
    function cleanup() {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    }
    function handleMessage(e) {
      const msg = e.data;
      if (msg.type === "progress") {
        onProgress(msg.progress);
      } else if (msg.type === "loaded") {
        cleanup();
        resolve();
      } else if (msg.type === "error") {
        cleanup();
        reject(new Error(msg.message));
      }
    }
    function handleError(e) {
      cleanup();
      reject(new Error(e.message || "Transformers.js worker crashed"));
    }
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ type: messageType, task, modelId, dtype, device });
  });
}

function workerGenerate(conversationHistory, onToken, messageType = "generate") {
  const worker = getTransformersWorker();
  return new Promise((resolve, reject) => {
    let full = "";
    function cleanup() {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    }
    function handleMessage(e) {
      const msg = e.data;
      if (msg.type === "token") {
        full += msg.text;
        onToken(full);
      } else if (msg.type === "done") {
        cleanup();
        resolve(full);
      } else if (msg.type === "error") {
        cleanup();
        reject(new Error(msg.message));
      }
    }
    function handleError(e) {
      cleanup();
      reject(new Error(e.message || "Transformers.js worker crashed"));
    }
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ type: messageType, history: conversationHistory, messages: conversationHistory });
  });
}

function workerCaption(file) {
  const worker = getTransformersWorker();
  return new Promise((resolve, reject) => {
    function cleanup() {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    }
    function handleMessage(e) {
      const msg = e.data;
      if (msg.type === "caption-result") {
        cleanup();
        resolve(msg.text);
      } else if (msg.type === "error") {
        cleanup();
        reject(new Error(msg.message));
      }
    }
    function handleError(e) {
      cleanup();
      reject(new Error(e.message || "Transformers.js worker crashed"));
    }
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ type: "caption", file });
  });
}

async function loadTransformersTask(task, loadingMessage, readyMessage, messageType = "load", device = undefined) {
  setLoadProgress(loadingMessage);
  const modelId = modelSelect.value;
  const { dtype } = getSelectedModelConfig();
  await withRetries(() =>
    workerLoad(task, modelId, dtype, transformersProgressCallback, messageType, device)
  );
  setLoadProgress(`Loaded ${modelId}. ${readyMessage}`, 100);
}

async function loadTransformers() {
  await loadTransformersTask("text-generation", "Loading Transformers.js...", "Ready to chat.");
  transformersPipeline = true;
}

async function loadCaptioner() {
  await loadTransformersTask(
    "image-to-text",
    "Loading image captioning model...",
    "Upload an image to caption it."
  );
  captionerPipeline = true;
}

async function loadVqa() {
  // Doesn't use the "image-text-to-text" pipeline task — confirmed against
  // the actual bundled source that the current @huggingface/transformers
  // release's pipeline() registry doesn't include it. The worker's
  // "load-vqa"/"vqa-generate" messages use the lower-level AutoProcessor +
  // AutoModelForVision2Seq API instead (see transformers-worker.js).
  //
  // Unlike the other Transformers.js providers, WebGPU isn't optional here:
  // tested on WASM/CPU and a single short reply took over 3 minutes without
  // finishing (vs. seconds for the text-only SmolLM2 provider) — the vision
  // encoder makes this model meaningfully heavier. Same requirement and
  // error pattern as loadWebLLM().
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this browser. Image Q&A needs it — WASM " +
        "alone is impractically slow for this model. Try a recent Chrome/Edge."
    );
  }
  await loadTransformersTask(
    "image-text-to-text",
    "Loading image Q&A model...",
    "Attach an image to start asking questions.",
    "load-vqa",
    "webgpu"
  );
  vqaPipeline = true;
}

async function loadChromeAI() {
  const LanguageModel = self.LanguageModel ?? self.ai?.languageModel;
  if (!LanguageModel) {
    throw new Error(
      "Chrome's built-in AI Prompt API is not available. Requires a recent " +
        "Chrome with chrome://flags/#prompt-api-for-gemini-nano enabled " +
        "(or an origin trial token)."
    );
  }

  const availability =
    (await LanguageModel.availability?.()) ?? "available";
  if (availability === "unavailable") {
    throw new Error(
      "Gemini Nano reports unavailable on this device/browser."
    );
  }

  setLoadProgress("Preparing Chrome built-in model...");
  chromeSession = await LanguageModel.create({
    monitor(m) {
      m.addEventListener("downloadprogress", (e) => {
        setLoadProgress(`Downloading Gemini Nano: ${Math.round(e.loaded * 100)}%`, e.loaded * 100);
      });
    },
  });
  setLoadProgress("Gemini Nano ready. Ready to chat.", 100);
}

const loaders = {
  webllm: loadWebLLM,
  transformers: loadTransformers,
  caption: loadCaptioner,
  vqa: loadVqa,
  chrome: loadChromeAI,
};

loadBtn.addEventListener("click", showConfirmModal);
composerHeroLoadBtn.addEventListener("click", showConfirmModal);

modalConfirmBtn.addEventListener("click", async () => {
  modalTitle.textContent = "Downloading model";
  modalConfirmSection.classList.add("hidden");
  modalProgressSection.classList.remove("hidden");
  modalProgressFill.classList.add("indeterminate");
  modalProgressFill.style.width = "0%";
  modalStatus.textContent = "Starting…";
  modalStatus.classList.remove("error");
  modalCloseBtn.classList.add("hidden");

  loadBtn.disabled = true;
  setChatEnabled(false);
  // Only the engine handles reset here, not the chat display — clicking
  // "Load model" to continue a restored session (see loadSession()) must
  // not wipe the conversation that's already on screen.
  resetEngineHandles();
  try {
    await loaders[providerSelect.value]();
    loadedModelKey = `${providerSelect.value}:${modelSelect.value}`;
    setChatEnabled(true);
    setTimeout(hideLoadModal, 500);
  } catch (err) {
    console.error(err);
    const message = err.message ?? String(err);
    setStatus(message, true);
    modalStatus.textContent = message;
    modalStatus.classList.add("error");
    modalProgressFill.classList.remove("indeterminate");
    modalCloseBtn.classList.remove("hidden");
  } finally {
    loadBtn.disabled = false;
  }
});

// Assistant replies are rendered as markdown (lists, code, bold, etc. from
// the model come through formatted instead of literal asterisks/backticks).
// User messages are never touched by this — they're always set via
// bubble.textContent elsewhere — so nothing a user types can execute as HTML.
//
// marked.js parses markdown but does not sanitize the resulting HTML, and
// local model output is not trustworthy input for innerHTML: a model asked
// to "show an example HTML page" without wrapping it in a code fence — a
// very plausible thing for a small/quantized instruct model to get wrong —
// would inject raw markup straight into the DOM. DOMPurify strips anything
// dangerous (script tags, event-handler attributes, javascript:/data: URIs,
// iframes, etc.) from marked's output before it's ever assigned to
// innerHTML. Both are needed together: marked doesn't sanitize, DOMPurify
// doesn't parse markdown.
let markedPromise = null;
function getMarked() {
  if (!markedPromise) {
    markedPromise = import("https://esm.run/marked").then((m) => m.marked);
  }
  return markedPromise;
}

let dompurifyPromise = null;
function getDOMPurify() {
  if (!dompurifyPromise) {
    dompurifyPromise = import("https://esm.run/dompurify").then((m) => m.default);
  }
  return dompurifyPromise;
}

async function renderAssistantMarkdown(bubble, text) {
  try {
    const [marked, DOMPurify] = await Promise.all([getMarked(), getDOMPurify()]);
    // If either library failed to load, we land in the catch below and fall
    // back to plain text — never to marked's unsanitized HTML. Fail-safe,
    // not fail-open.
    bubble.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } catch (err) {
    console.error("Markdown rendering failed, showing plain text", err);
    bubble.textContent = text;
  }
}

async function streamWebLLM(bubble) {
  const chunks = await webllmEngine.chat.completions.create({
    messages: history,
    stream: true,
  });
  let full = "";
  for await (const chunk of chunks) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    full += delta;
    await renderAssistantMarkdown(bubble, full);
    chatEl.scrollTop = chatEl.scrollHeight;
  }
  return full;
}

async function streamTransformers(bubble) {
  return workerGenerate(history, (fullText) => {
    renderAssistantMarkdown(bubble, fullText);
    chatEl.scrollTop = chatEl.scrollHeight;
  });
}

// VQA messages mirror the text-chat {role, content} shape everywhere except
// the very first user turn, which embeds the attached image alongside the
// question — the worker's "generate" handler just forwards whatever shape
// it's given straight into the pipeline, so no worker changes were needed.
function buildVqaPayload() {
  return vqaMessages.map((msg, i) =>
    i === 0
      ? { role: msg.role, content: [{ type: "image", image: vqaImage }, { type: "text", text: msg.content }] }
      : { role: msg.role, content: msg.content }
  );
}

async function streamVqa(bubble) {
  return workerGenerate(
    buildVqaPayload(),
    (fullText) => {
      renderAssistantMarkdown(bubble, fullText);
      chatEl.scrollTop = chatEl.scrollHeight;
    },
    "vqa-generate"
  );
}

async function streamChromeAI(bubble) {
  const lastUserMessage = history[history.length - 1].content;
  const stream = chromeSession.promptStreaming(lastUserMessage);
  let full = "";
  for await (const chunk of stream) {
    // Some Chrome versions yield cumulative text, others yield deltas.
    if (chunk.startsWith(full)) {
      full = chunk;
    } else {
      full += chunk;
    }
    await renderAssistantMarkdown(bubble, full);
    chatEl.scrollTop = chatEl.scrollHeight;
  }
  return full;
}

const streamers = {
  webllm: streamWebLLM,
  transformers: streamTransformers,
  vqa: streamVqa,
  chrome: streamChromeAI,
};

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;
  imageInput.value = ""; // allow re-selecting the same file later

  setChatEnabled(false);
  addImageBubble(file);

  const bubble = addBubble("assistant");
  bubble.classList.add("pending");

  try {
    const caption = await workerCaption(file);
    bubble.textContent = caption || "(no caption produced)";
  } catch (err) {
    console.error(err);
    bubble.textContent = `Error: ${err.message ?? err}`;
    setStatus(err.message ?? String(err), true);
  } finally {
    bubble.classList.remove("pending");
    setChatEnabled(true);
  }
});

// Attaching an image in VQA mode doesn't generate anything by itself — it
// just starts (or restarts) the image conversation and unlocks the text
// input, mirroring "Attach an image to start…" in the placeholder.
vqaImageInput.addEventListener("change", () => {
  const file = vqaImageInput.files[0];
  if (!file) return;
  vqaImageInput.value = "";

  vqaImage = file;
  vqaMessages = [];
  chatInner.innerHTML = "";
  addImageBubble(file);
  updateEmptyState();

  input.placeholder = "Ask about this image...";
  setChatEnabled(true);
  input.focus();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  const isVqa = providerSelect.value === "vqa";

  input.value = "";
  setChatEnabled(false);

  const targetHistory = isVqa ? vqaMessages : history;
  targetHistory.push({ role: "user", content: text });
  addBubble("user").textContent = text;

  const bubble = addBubble("assistant");
  bubble.classList.add("pending");

  try {
    const reply = await streamers[providerSelect.value](bubble);
    targetHistory.push({ role: "assistant", content: reply });
  } catch (err) {
    console.error(err);
    bubble.textContent = `Error: ${err.message ?? err}`;
    setStatus(err.message ?? String(err), true);
  } finally {
    bubble.classList.remove("pending");
    setChatEnabled(true);
    input.focus();
    // VQA sessions include an in-memory image and aren't persisted to
    // localStorage — see the note on `vqaImage`/`vqaMessages` above.
    if (!isVqa) upsertCurrentSession();
  }
});
