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
    // The four below were checked against the live prebuiltAppConfig.model_list
    // this app actually loads (CreateMLCEngine resolves modelId against that
    // list, so an unlisted id fails outright) — confirmed present, and
    // sizeEstimate is that config's real vram_required_MB, not a guess.
    { value: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2 360M Instruct (q4f16)", sizeEstimate: "~376 MB" },
    { value: "Qwen3-0.6B-q4f16_1-MLC", label: "Qwen 3 0.6B (q4f16)", sizeEstimate: "~1.4 GB" },
    { value: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 1.5B Instruct (q4f16)", sizeEstimate: "~1.6 GB" },
    // f32-activation build for GPUs that lack the shader-f16 feature the
    // q4f16 entries above require (some Android GPUs) — same weights as the
    // Llama entry above, just a variant that doesn't need shader-f16.
    { value: "Llama-3.2-1B-Instruct-q4f32_1-MLC", label: "Llama 3.2 1B Instruct (q4f32, no shader-f16 needed)", sizeEstimate: "~1.13 GB" },
  ],
  transformers: [
    { value: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen 2.5 0.5B Instruct (ONNX, q4)", dtype: "q4", sizeEstimate: "~300 MB" },
    { value: "HuggingFaceTB/SmolLM2-135M-Instruct", label: "SmolLM2 135M Instruct (ONNX, very small/fast)", sizeEstimate: "~130 MB" },
    // Confirmed end-to-end (load + real generation) via headless Chromium's
    // WASM path. sizeEstimate is the real onnx/model_q4.onnx_data size from
    // HF, not a guess (a candidate estimate of ~230MB floating around
    // elsewhere was wrong — the real q4 weights are ~388MB).
    { value: "HuggingFaceTB/SmolLM2-360M-Instruct", label: "SmolLM2 360M Instruct (ONNX, q4)", dtype: "q4", sizeEstimate: "~390 MB" },
  ],
  caption: [
    // The default quantized export of this model has a broken decoder graph
    // ("Missing required scale" DequantizeLinear error) under current ONNX
    // Runtime Web — fp32 sidesteps it at the cost of a bigger download.
    { value: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Image Captioning", dtype: "fp32", sizeEstimate: "~800 MB" },
  ],
  vqa: [
    { value: "HuggingFaceTB/SmolVLM-500M-Instruct", label: "SmolVLM 500M Instruct (image Q&A)", dtype: "q4", sizeEstimate: "~340 MB" },
    // Same family/architecture as the 500M model above (just the smaller
    // variant), confirmed the processor+model construct without throwing at
    // this dtype — the failure mode that's actually bitten this app before
    // (broken quantized graphs, unsupported ops). Not verified beyond that:
    // an actual image Q&A exchange on real hardware, the way the 500M model
    // above was verified, still hasn't been done for this one.
    { value: "HuggingFaceTB/SmolVLM-256M-Instruct", label: "SmolVLM 256M Instruct (image Q&A, smallest)", dtype: "q4", sizeEstimate: "~265 MB" },
  ],
  chrome: [],
  // Not a selectable provider in the #provider dropdown — voice input is
  // orthogonal to which chat provider is active, so this only exists as a
  // config home (reused by allCacheableModels() for the storage inspector,
  // same as every other entry above) and isn't ever assigned to modelSelect.
  asr: [
    // Same broken-default-quantized-graph issue as the vit-gpt2 captioning
    // model above ("Missing required scale" DequantizeLinear error) —
    // dtype: "fp32" sidesteps it. Confirmed via direct testing: the default
    // dtype throws on session creation, fp32 loads (~5-7s) and transcribes
    // correctly. sizeEstimate is the real onnx/ file sizes from HF (encoder
    // 32.9MB + decoder 118.4MB), not a guess — quantized would be ~41MB but
    // is unusable.
    { value: "Xenova/whisper-tiny.en", label: "Whisper Tiny English (speech-to-text)", dtype: "fp32", sizeEstimate: "~150 MB" },
  ],
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
const stopBtn = document.getElementById("stop-btn");
const imageInput = document.getElementById("image-input");
const imageLabel = document.getElementById("image-label");
const vqaImageInput = document.getElementById("vqa-image-input");
const vqaAttachLabel = document.getElementById("vqa-attach-label");
const micBtn = document.getElementById("mic-btn");
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
const menuCapabilitiesBtn = document.getElementById("menu-capabilities-btn");
const menuCloseBtn = document.getElementById("menu-close-btn");
const aboutModal = document.getElementById("about-modal");
const aboutCloseBtn = document.getElementById("about-close-btn");
const capabilitiesModal = document.getElementById("capabilities-modal");
const capabilitiesSummary = document.getElementById("capabilities-summary");
const capabilitiesList = document.getElementById("capabilities-list");
const capabilitiesProviders = document.getElementById("capabilities-providers");
const capabilitiesCloseBtn = document.getElementById("capabilities-close-btn");
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

// Voice input state. asrLoaded is deliberately separate from loadedModelKey
// — the ASR pipeline's own dedicated worker (asrWorker) is independent of
// whichever chat provider is currently loaded, so it must survive provider
// switches and must never be touched by resetEngineHandles(). pendingLoadKind
// tells the shared confirm/progress modal (showConfirmModal(), the
// modalConfirmBtn handler) which of the two unrelated things it's currently
// downloading: a chat-provider model, or the voice-input model.
let asrWorker = null;
let asrLoaded = false;
let isRecording = false;
let mediaRecorder = null;
let micStream = null;
let recordedChunks = [];
let pendingLoadKind = "provider";
let pendingLoadCached = false;

// "Streaming" voice input: Whisper itself isn't a streaming model, so this
// fakes the natural feel by re-transcribing the growing recording every
// PARTIAL_TRANSCRIBE_INTERVAL_MS and pushing the latest guess into the input
// as the user talks, instead of only filling it once at the end. Each pass
// re-runs on the whole clip so far (there's no cheaper incremental Whisper
// API), so it costs more compute the longer someone talks — fine for the
// short, few-sentence utterances this is meant for. isTranscribingPartial
// skips a tick if the previous pass hasn't finished rather than queuing one,
// so a slow (e.g. WASM-only) device just gets less frequent updates rather
// than a growing backlog.
const PARTIAL_TRANSCRIBE_INTERVAL_MS = 1500;
let partialTranscribeTimer = null;
let isTranscribingPartial = false;

// Provider-specific "stop generating" hook, set by each stream*() function
// right before it starts and cleared once it finishes — there's only ever
// one generation in flight at a time. userStoppedGeneration distinguishes a
// user-requested stop from a real mid-generation failure, since WebLLM and
// Chrome's Prompt API both surface an interruption as a thrown/rejected
// error rather than a clean early return.
let activeStop = null;
let userStoppedGeneration = false;

// Incremented once per generated chunk by whichever stream*() is currently
// running (one WebLLM/Chrome AI stream chunk, one Transformers.js "token"
// worker message) — a proxy for token count, not a real tokenizer count, but
// close enough for a rough tok/s readout under each reply. Reset per
// generation by beginGenerating() so a stale count from a prior reply can't
// leak into the next one's stats.
let streamChunkCount = 0;

function beginGenerating() {
  userStoppedGeneration = false;
  streamChunkCount = 0;
  sendBtn.classList.add("hidden");
  stopBtn.classList.remove("hidden");
}

function endGenerating() {
  activeStop = null;
  sendBtn.classList.remove("hidden");
  stopBtn.classList.add("hidden");
}

stopBtn.addEventListener("click", () => {
  userStoppedGeneration = true;
  activeStop?.();
});

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

// cached=false is always the first paint (no network round-trip before the
// modal can open); showConfirmModal() below kicks off an async cache check
// right after and re-calls this to correct the copy to "Load" in place if
// the model turns out to already be downloaded.
function applyConfirmModalCopy({ modelLabel, cached, isChrome, config }) {
  pendingLoadCached = cached;
  modalTitle.textContent = cached ? "Load model?" : "Download model?";
  modalModelName.textContent = modelLabel;
  modalSizeWarning.textContent = isChrome
    ? "Chrome manages this download itself — size and progress aren't reported to this page."
    : cached
      ? "Already downloaded and cached in this browser — this will load it from cache, no network needed."
      : `Estimated download size: ${config?.sizeEstimate ?? "unknown"} (approximate).`;
  modalConfirmBtn.textContent = cached ? "Load" : "Download";
}

function showConfirmModal(kind = "provider") {
  pendingLoadKind = kind;
  const config = kind === "asr" ? getAsrModelConfig() : getSelectedModelConfig();
  const modelLabel =
    config?.label ?? providerSelect.options[providerSelect.selectedIndex].textContent;
  const isChrome = kind !== "asr" && providerSelect.value === "chrome";

  applyConfirmModalCopy({ modelLabel, cached: false, isChrome, config });

  modalConfirmSection.classList.remove("hidden");
  modalProgressSection.classList.add("hidden");
  loadModal.classList.remove("hidden");

  if (!isChrome && config && window.caches) {
    findCachedEntriesFor(config.value).then((entries) => {
      const stillRelevant = pendingLoadKind === kind && !loadModal.classList.contains("hidden");
      if (entries.length && stillRelevant) {
        applyConfirmModalCopy({ modelLabel, cached: true, isChrome, config });
      }
    });
  }
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

// Confirmed via real on-device testing (iOS Safari, WebGPU available once
// excluded from Lockdown Mode's JIT restriction): WebLLM's Qwen2.5 0.5B is
// the fastest option tried so far on mobile, beating the WASM-only
// Transformers.js path this used to default to. Only applies to the initial
// dropdown selection on a fresh load; explicit choices (a provider/model
// switch, or restoring a saved session) always win.
const mobileDefaultModel = {
  provider: "webllm",
  modelValue: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
};

const providerLabels = {
  webllm: "WebLLM",
  transformers: "Transformers.js",
  caption: "Transformers.js (captioning)",
  vqa: "Transformers.js (image Q&A)",
  chrome: "Chrome built-in AI",
  asr: "Voice input (Whisper)",
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
        markDownloadedModels(providerSelect.value);
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
  markDownloadedModels(providerSelect.value);
});

// Gathers everything the "Capabilities" panel shows in one pass, so both the
// raw device-info rows and the per-provider support rows are derived from a
// single set of API calls rather than re-querying navigator.gpu/LanguageModel
// twice. Read-only — this doesn't load or touch any provider.
async function computeCapabilities() {
  const webgpu = { supported: !!navigator.gpu, description: null };
  if (webgpu.supported) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter?.info ?? (await adapter?.requestAdapterInfo?.());
      const parts = [info?.vendor, info?.architecture, info?.description].filter(Boolean);
      webgpu.description = parts.length
        ? parts.join(" · ")
        : adapter
          ? "Available (details not exposed by this browser)"
          : "Adapter request failed";
    } catch {
      webgpu.description = "Available (details not exposed by this browser)";
    }
  }

  // WebGPU adapter info is often deliberately withheld (anti-fingerprinting),
  // so fall back to the classic WebGL "unmasked renderer" string, which many
  // browsers still expose and which frequently names the actual GPU model.
  let glRenderer = null;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    glRenderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
  } catch {
    glRenderer = null;
  }

  const LanguageModel = self.LanguageModel ?? self.ai?.languageModel;
  const geminiNano = { supported: !!LanguageModel, availability: null };
  if (LanguageModel) {
    try {
      geminiNano.availability = (await LanguageModel.availability?.()) ?? "available";
    } catch {
      geminiNano.availability = "unavailable";
    }
  }

  const storage = navigator.storage?.estimate
    ? await navigator.storage.estimate()
    : null;

  return {
    webgpu,
    glRenderer,
    cpuCores: navigator.hardwareConcurrency ?? null,
    deviceMemory: navigator.deviceMemory ?? null,
    wasm: typeof WebAssembly !== "undefined",
    geminiNano,
    storage,
  };
}

function providerSupport(caps) {
  const geminiOk = caps.geminiNano.supported && caps.geminiNano.availability !== "unavailable";
  return {
    webllm: { ok: caps.webgpu.supported, reason: caps.webgpu.supported ? "Supported" : "Requires WebGPU" },
    transformers: { ok: caps.wasm, reason: "Supported (WebGPU if available, WASM otherwise)" },
    caption: { ok: caps.wasm, reason: "Supported (WebGPU if available, WASM otherwise)" },
    vqa: { ok: caps.webgpu.supported, reason: caps.webgpu.supported ? "Supported (requires WebGPU)" : "Requires WebGPU" },
    chrome: { ok: geminiOk, reason: geminiOk ? `Available (${caps.geminiNano.availability})` : "Not available" },
  };
}

// status is "ok" | "warn" | "bad" | undefined (undefined = neutral, for
// purely informational rows with no pass/fail meaning, e.g. CPU core count).
function capabilityRow(name, meta, { wide = false, status } = {}) {
  const el = document.createElement("div");
  el.className = wide ? "settings-row settings-row--wide" : "settings-row";
  if (status) el.classList.add(`status-${status}`);
  const info = document.createElement("div");
  const nameEl = document.createElement("div");
  nameEl.className = "settings-row-name";
  nameEl.textContent = name;
  const metaEl = document.createElement("div");
  metaEl.className = "settings-row-meta";
  metaEl.textContent = meta;
  info.append(nameEl, metaEl);
  el.appendChild(info);
  return el;
}

async function renderCapabilities() {
  capabilitiesSummary.textContent = "Checking this browser/device...";
  capabilitiesList.innerHTML = "";
  capabilitiesProviders.innerHTML = "";

  const caps = await computeCapabilities();
  capabilitiesSummary.textContent =
    "Everything below is detected locally in this browser — nothing is sent anywhere.";

  const gpuMeta = caps.webgpu.supported
    ? caps.webgpu.description
    : caps.glRenderer
      ? `Not available (WebGL reports: ${caps.glRenderer})`
      : "Not available in this browser";
  capabilitiesList.appendChild(
    capabilityRow("WebGPU", gpuMeta, { wide: true, status: caps.webgpu.supported ? "ok" : "bad" })
  );
  if (caps.webgpu.supported && caps.glRenderer) {
    capabilitiesList.appendChild(capabilityRow("GPU (WebGL renderer)", caps.glRenderer, { wide: true }));
  }
  capabilitiesList.appendChild(
    capabilityRow("CPU cores", caps.cpuCores ? String(caps.cpuCores) : "Not reported by this browser")
  );
  capabilitiesList.appendChild(
    capabilityRow(
      "Device memory",
      caps.deviceMemory ? `~${caps.deviceMemory} GB (approx, Chrome/Edge-only API)` : "Not reported by this browser"
    )
  );
  capabilitiesList.appendChild(
    capabilityRow("WebAssembly", caps.wasm ? "Supported" : "Not supported", { status: caps.wasm ? "ok" : "bad" })
  );
  const geminiStatus = !caps.geminiNano.supported
    ? "bad"
    : caps.geminiNano.availability === "available"
      ? "ok"
      : caps.geminiNano.availability === "unavailable"
        ? "bad"
        : "warn"; // e.g. "downloadable" / "downloading" — usable soon, not yet
  capabilitiesList.appendChild(
    capabilityRow(
      "Gemini Nano (Chrome built-in AI)",
      caps.geminiNano.supported ? `Availability: ${caps.geminiNano.availability}` : "Not supported in this browser",
      { status: geminiStatus }
    )
  );
  const storageRatio = caps.storage?.quota ? caps.storage.usage / caps.storage.quota : null;
  capabilitiesList.appendChild(
    capabilityRow(
      "Storage quota",
      caps.storage?.quota
        ? `Using ${formatBytes(caps.storage.usage)} of roughly ${formatBytes(caps.storage.quota)} available`
        : "Not reported by this browser",
      { status: storageRatio !== null && storageRatio > 0.9 ? "warn" : undefined }
    )
  );

  const support = providerSupport(caps);
  for (const [provider, { ok, reason }] of Object.entries(support)) {
    capabilitiesProviders.appendChild(
      capabilityRow(providerLabels[provider], `${ok ? "✓" : "✗"} ${reason}`, { status: ok ? "ok" : "bad" })
    );
  }
}

// The gear icon opens a small "links" modal (Storage / About / Capabilities)
// rather than a dropdown, matching the same modal-overlay pattern used
// everywhere else.
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

function openCapabilitiesModal() {
  capabilitiesModal.classList.remove("hidden");
  renderCapabilities();
}

function closeCapabilitiesModal() {
  capabilitiesModal.classList.add("hidden");
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
menuCapabilitiesBtn.addEventListener("click", () => {
  closeMenuModal();
  openCapabilitiesModal();
});

aboutCloseBtn.addEventListener("click", closeAboutModal);
aboutModal.addEventListener("click", (e) => {
  if (e.target === aboutModal) closeAboutModal();
});

capabilitiesCloseBtn.addEventListener("click", closeCapabilitiesModal);
capabilitiesModal.addEventListener("click", (e) => {
  if (e.target === capabilitiesModal) closeCapabilitiesModal();
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!aboutModal.classList.contains("hidden")) closeAboutModal();
  else if (!capabilitiesModal.classList.contains("hidden")) closeCapabilitiesModal();
  else if (!menuModal.classList.contains("hidden")) closeMenuModal();
});

function setChatEnabled(enabled) {
  // In VQA mode, the text input additionally requires an image to already be
  // attached for the current session — nothing to ask a question about yet.
  const vqaReady = providerSelect.value !== "vqa" || vqaImage !== null;
  input.disabled = !enabled || !vqaReady;
  sendBtn.disabled = !enabled || !vqaReady;
  // Mic fills the same text input, so it's confusing to let voice input
  // work while typing/sending don't — gate it identically, even though the
  // ASR model it actually uses is otherwise independent of the chat engine.
  micBtn.disabled = !enabled || !vqaReady;
  // A greyed-out icon button with no explanation is easy to mistake for
  // broken rather than disabled-for-a-reason — spell out why via the same
  // title/aria-label used for the default and recording states.
  micBtn.title = !enabled
    ? "Load a model first"
    : !vqaReady
      ? "Attach an image first"
      : "Voice input";
  micBtn.setAttribute(
    "aria-label",
    micBtn.disabled ? micBtn.title : isRecording ? "Stop recording" : "Start voice input"
  );
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

  // .msg-col wraps the bubble so a .msg-stats line (tok/s, added after an
  // assistant reply finishes — see setMsgStats()) can sit directly under it
  // without disturbing the row's avatar+bubble flex layout.
  const col = document.createElement("div");
  col.className = "msg-col";

  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;

  col.appendChild(bubble);
  row.append(avatar, col);
  chatInner.appendChild(row);
  updateEmptyState();
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

// Small "12.3 tok/s · 42 tokens · 3.4s" line under an assistant reply once
// generation finishes. chunkCount is a proxy for real token count (see
// streamChunkCount above) — approximate, not a tokenizer-verified number.
function setMsgStats(bubble, chunkCount, elapsedSec) {
  if (chunkCount <= 0 || elapsedSec <= 0) return;
  const stats = document.createElement("div");
  stats.className = "msg-stats";
  stats.textContent = `${(chunkCount / elapsedSec).toFixed(1)} tok/s · ${chunkCount} tokens · ${elapsedSec.toFixed(1)}s`;
  bubble.parentElement.appendChild(stats);
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
  markDownloadedModels(provider);
}

// Subtle "already downloaded" indicator on the model dropdown, so switching
// providers/models shows which options avoid a fresh multi-hundred-MB
// download. Reuses the same Cache API scan the storage inspector already
// does (findCachedEntriesFor) rather than tracking download state ourselves.
// Runs after populateModelOptions rebuilds the <option> list, and again
// after a load completes or a cache entry is cleared, since either can
// change which models are actually cached.
async function markDownloadedModels(provider) {
  if (!window.caches) return;
  const models = providerModels[provider];
  const options = Array.from(modelSelect.options);
  await Promise.all(
    models.map(async ({ value, label }, i) => {
      const entries = await findCachedEntriesFor(value);
      const opt = options[i];
      if (opt) opt.textContent = entries.length ? `${label} ✓` : label;
    })
  );
}

function getSelectedModelConfig() {
  return providerModels[providerSelect.value].find(
    (m) => m.value === modelSelect.value
  );
}

function getAsrModelConfig() {
  return providerModels.asr[0];
}

// Mic button shows wherever there's a free-text input to fill — every
// provider except captioning, which has no text input at all. Also hidden
// on mobile: transcription was too slow to be usable there in practice
// (reported directly, not a hypothetical — likely mobile CPU being too slow
// for Whisper's WASM path), so there's no point offering it. Kept separate
// from syncInputMode() (rather than folded into it) so a viewport resize can
// resync just this without also resetting input.placeholder, which would
// clobber "Ask about this image..." if a VQA image is already attached.
function updateMicVisibility(provider) {
  micBtn.classList.toggle("hidden", provider === "caption" || isMobileViewport());
}

function syncInputMode(provider) {
  const isCaption = provider === "caption";
  textControls.classList.toggle("hidden", isCaption);
  imageControls.classList.toggle("hidden", !isCaption);
  vqaAttachLabel.classList.toggle("hidden", provider !== "vqa");
  updateMicVisibility(provider);
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

if (isMobileViewport()) providerSelect.value = mobileDefaultModel.provider;
populateModelOptions(providerSelect.value);
if (isMobileViewport()) modelSelect.value = mobileDefaultModel.modelValue;
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
// Same resync need applies to the mic button's mobile-hidden state.
window.addEventListener("resize", () => {
  setSidebarOpen(!sidebar.classList.contains("collapsed"));
  updateMicVisibility(providerSelect.value);
});

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
const WORKER_VERSION = "3";

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

// Set for the duration of a workerGenerate() call so hardStopWorkerGeneration
// can settle its promise early — Worker.terminate() doesn't fire any event on
// the main thread, so without this the promise would otherwise just hang.
let pendingWorkerGeneration = null;

// Stopping a Transformers.js/VQA generation in progress needs the worker
// killed outright (see the note in transformers-worker.js on why a graceful
// postMessage-based interrupt doesn't work here) — which also means the
// loaded model is gone and has to be reloaded before the next message.
// loadedModelKey is cleared for exactly that reason: it's the same signal
// loadSession() already uses to show "load this model to continue."
function hardStopWorkerGeneration() {
  const full = pendingWorkerGeneration?.full ?? "";
  const resolve = pendingWorkerGeneration?.resolve;
  pendingWorkerGeneration = null;
  terminateTransformersWorker();
  loadedModelKey = null;
  setStatus("Stopped. Load the model again to continue chatting.");
  resolve?.(full);
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
    const state = { full: "", resolve };
    pendingWorkerGeneration = state;
    function cleanup() {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      pendingWorkerGeneration = null;
    }
    function handleMessage(e) {
      const msg = e.data;
      if (msg.type === "token") {
        state.full += msg.text;
        onToken(state.full);
      } else if (msg.type === "done") {
        cleanup();
        resolve(state.full);
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

// Voice input runs in its own dedicated worker (asr-worker.js), not
// transformersWorker above — that one holds exactly one loaded pipeline at a
// time and is torn down on every chat-provider switch (resetEngineHandles()),
// which would be wrong here: voice input must keep working (and stay loaded)
// regardless of which chat provider is currently selected, including WebLLM,
// which doesn't use transformersWorker at all. Same cache-busting convention
// as WORKER_VERSION above — bump whenever asr-worker.js changes.
const ASR_WORKER_VERSION = "2";

function getAsrWorker() {
  if (!asrWorker) {
    asrWorker = new Worker(`asr-worker.js?v=${ASR_WORKER_VERSION}`, { type: "module" });
  }
  return asrWorker;
}

function asrWorkerLoad(modelId, dtype, onProgress) {
  const worker = getAsrWorker();
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
      reject(new Error(e.message || "Voice input worker crashed"));
    }
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ type: "load", modelId, dtype });
  });
}

// Whisper's audio decoding needs AudioContext, which doesn't exist inside a
// Worker (confirmed via direct testing: passing a URL for the ASR worker to
// fetch+decode itself throws "AudioContext is not available in your
// environment") — so decoding has to happen here, on the main thread, before
// handing the result to the worker. read_audio() does the decode+resample to
// 16kHz in one step; the pipeline itself accepts the resulting Float32Array
// directly (confirmed — unlike a raw Blob, which it rejects).
async function decodeAudioBlob(blob) {
  const { read_audio } = await import("https://esm.run/@huggingface/transformers");
  const url = URL.createObjectURL(blob);
  try {
    return await read_audio(url, 16000);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function asrWorkerTranscribe(audioData) {
  const worker = getAsrWorker();
  return new Promise((resolve, reject) => {
    function cleanup() {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    }
    function handleMessage(e) {
      const msg = e.data;
      if (msg.type === "transcript") {
        cleanup();
        resolve(msg.text);
      } else if (msg.type === "error") {
        cleanup();
        reject(new Error(msg.message));
      }
    }
    function handleError(e) {
      cleanup();
      reject(new Error(e.message || "Voice input worker crashed"));
    }
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({ type: "transcribe", audioData });
  });
}

// asrWorkerTranscribe() attaches a fresh "message" listener per call to the
// one shared asrWorker — fine when only a single call is ever in flight, but
// the live partial-transcribe loop (runPartialTranscribe(), below) and the
// final pass on stop (handleRecordingStopped()) can now legitimately overlap
// (a partial tick still resolving when the user releases the mic). Two
// listeners on the same worker would both race to consume whichever
// "transcript" message arrives first, resolving the wrong caller's promise.
// Routing every transcribe call through this queue guarantees only one is
// ever in flight, so each caller's listener sees exactly its own reply.
let asrTranscribeQueue = Promise.resolve();
function queuedAsrTranscribe(audioData) {
  const run = asrTranscribeQueue.then(() => asrWorkerTranscribe(audioData));
  asrTranscribeQueue = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

function asrProgressCallback(progress) {
  if (progress.status === "progress") {
    const pct = Math.round(progress.progress ?? 0);
    setLoadProgress(`Downloading ${progress.file} (${pct}%)...`, pct);
  } else {
    setLoadProgress(`${progress.status}...`);
  }
}

async function loadAsrModel() {
  const { value: modelId, dtype } = getAsrModelConfig();
  setLoadProgress("Loading voice input model...");
  await withRetries(() => asrWorkerLoad(modelId, dtype, asrProgressCallback));
  setLoadProgress(`Loaded ${modelId}. Ready to transcribe.`, 100);
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

// showConfirmModal() now takes an optional kind param ("provider" | "asr") —
// these two listeners used to pass it directly, but a DOM click handler's
// first argument is the click Event, not undefined, so that would have
// silently overridden the "provider" default. Wrap them instead.
loadBtn.addEventListener("click", () => showConfirmModal());
composerHeroLoadBtn.addEventListener("click", () => showConfirmModal());

modalConfirmBtn.addEventListener("click", async () => {
  modalTitle.textContent = pendingLoadCached ? "Loading model" : "Downloading model";
  modalConfirmSection.classList.add("hidden");
  modalProgressSection.classList.remove("hidden");
  modalProgressFill.classList.add("indeterminate");
  modalProgressFill.style.width = "0%";
  modalStatus.textContent = "Starting…";
  modalStatus.classList.remove("error");
  modalCloseBtn.classList.add("hidden");

  if (pendingLoadKind === "asr") {
    micBtn.disabled = true;
    try {
      await loadAsrModel();
      asrLoaded = true;
      setTimeout(() => {
        hideLoadModal();
        // The user clicked the mic wanting to talk, not to watch a download
        // finish — start listening immediately rather than making them
        // click a second time for the one thing they asked for.
        startRecording();
      }, 500);
    } catch (err) {
      console.error(err);
      const message = err.message ?? String(err);
      setStatus(message, true);
      modalStatus.textContent = message;
      modalStatus.classList.add("error");
      modalProgressFill.classList.remove("indeterminate");
      modalCloseBtn.classList.remove("hidden");
    } finally {
      micBtn.disabled = false;
    }
    return;
  }

  loadBtn.disabled = true;
  setChatEnabled(false);
  // Only the engine handles reset here, not the chat display — clicking
  // "Load model" to continue a restored session (see loadSession()) must
  // not wipe the conversation that's already on screen. Also deliberately
  // does not touch asrWorker/asrLoaded — voice input's lifecycle is
  // independent of the chat-provider selection (see the note by
  // ASR_WORKER_VERSION above).
  resetEngineHandles();
  try {
    await loaders[providerSelect.value]();
    loadedModelKey = `${providerSelect.value}:${modelSelect.value}`;
    setChatEnabled(true);
    markDownloadedModels(providerSelect.value);
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
  activeStop = () => webllmEngine.interruptGenerate();
  let full = "";
  try {
    const chunks = await webllmEngine.chat.completions.create({
      messages: history,
      stream: true,
    });
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content ?? "";
      if (delta) streamChunkCount++;
      full += delta;
      await renderAssistantMarkdown(bubble, full);
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  } catch (err) {
    // interruptGenerate() can surface as a rejection depending on exactly
    // when it lands relative to the current chunk — a user-requested stop
    // isn't a real failure, so return what streamed so far instead of
    // showing an error bubble.
    if (!userStoppedGeneration) throw err;
  }
  return full;
}

async function streamTransformers(bubble) {
  activeStop = hardStopWorkerGeneration;
  return workerGenerate(history, (fullText) => {
    streamChunkCount++;
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
  activeStop = hardStopWorkerGeneration;
  return workerGenerate(
    buildVqaPayload(),
    (fullText) => {
      streamChunkCount++;
      renderAssistantMarkdown(bubble, fullText);
      chatEl.scrollTop = chatEl.scrollHeight;
    },
    "vqa-generate"
  );
}

async function streamChromeAI(bubble) {
  const controller = new AbortController();
  activeStop = () => controller.abort();
  const lastUserMessage = history[history.length - 1].content;
  let full = "";
  try {
    const stream = chromeSession.promptStreaming(lastUserMessage, { signal: controller.signal });
    for await (const chunk of stream) {
      streamChunkCount++;
      // Some Chrome versions yield cumulative text, others yield deltas.
      if (chunk.startsWith(full)) {
        full = chunk;
      } else {
        full += chunk;
      }
      await renderAssistantMarkdown(bubble, full);
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  } catch (err) {
    // AbortError from the signal firing — the for-await loop throws it
    // rather than just ending, per Chrome's documented Prompt API behavior.
    if (!userStoppedGeneration) throw err;
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

// Voice input. The mic technically only needs its own (independent) ASR
// model, not a loaded chat engine — but its whole purpose is filling the
// text input, so it's wired into setChatEnabled() to stay disabled whenever
// that input is (see above). It's also briefly disabled for the duration of
// its own load/transcribe calls, to prevent double-clicks.
function setMicRecordingUI(recording) {
  micBtn.classList.toggle("recording", recording);
  micBtn.setAttribute("aria-label", recording ? "Stop recording" : "Start voice input");
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    setStatus("Voice input isn't supported in this browser.", true);
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus(
      err.name === "NotAllowedError"
        ? "Microphone access was denied. Allow microphone access in your browser settings to use voice input."
        : `Couldn't access the microphone: ${err.message ?? err}`,
      true
    );
    return;
  }
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(micStream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecordingStopped;
  // A timeslice makes dataavailable fire every 250ms instead of only once at
  // stop() — recordedChunks then already has audio to work with as soon as
  // the first partial-transcribe tick below fires.
  mediaRecorder.start(250);
  isRecording = true;
  setMicRecordingUI(true);
  setStatus("Listening…");
  input.classList.add("interim-transcript");
  partialTranscribeTimer = setInterval(runPartialTranscribe, PARTIAL_TRANSCRIBE_INTERVAL_MS);
}

function stopRecording() {
  clearInterval(partialTranscribeTimer);
  partialTranscribeTimer = null;
  mediaRecorder?.stop(); // triggers onstop -> handleRecordingStopped
  micStream?.getTracks().forEach((t) => t.stop()); // release the mic/tab indicator
  isRecording = false;
  setMicRecordingUI(false);
}

// Runs on a timer while recording, re-transcribing everything captured so
// far and updating the input live so text fills in as the user talks rather
// than only appearing once they stop. Best-effort: errors here are logged
// and swallowed rather than surfaced, since the authoritative transcript is
// the final pass in handleRecordingStopped() once recording actually ends.
async function runPartialTranscribe() {
  if (isTranscribingPartial || recordedChunks.length === 0) return;
  isTranscribingPartial = true;
  try {
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    if (blob.size > 0) {
      const audioData = await decodeAudioBlob(blob);
      const text = await queuedAsrTranscribe(audioData);
      if (isRecording) input.value = text.trim();
    }
  } catch (err) {
    console.warn("Partial transcription failed:", err);
  } finally {
    isTranscribingPartial = false;
  }
}

async function handleRecordingStopped() {
  const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
  recordedChunks = [];
  input.classList.remove("interim-transcript");
  if (blob.size === 0) {
    setStatus("No audio captured — try again.", true);
    return;
  }
  setStatus("Transcribing…");
  micBtn.disabled = true;
  try {
    const audioData = await decodeAudioBlob(blob);
    const text = await queuedAsrTranscribe(audioData);
    // Fills the input for the user to review/edit and send themselves —
    // never auto-submitted (no form.requestSubmit()/.submit() call here).
    input.value = text.trim();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(`Transcription failed: ${err.message ?? err}`, true);
  } finally {
    micBtn.disabled = false;
  }
}

async function handleMicClick() {
  if (!asrLoaded) {
    showConfirmModal("asr");
    return;
  }
  if (isRecording) stopRecording();
  else await startRecording();
}

micBtn.addEventListener("click", handleMicClick);

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

  beginGenerating();
  const startTime = performance.now();
  try {
    const reply = await streamers[providerSelect.value](bubble);
    targetHistory.push({ role: "assistant", content: reply });
    setMsgStats(bubble, streamChunkCount, (performance.now() - startTime) / 1000);
  } catch (err) {
    console.error(err);
    bubble.textContent = `Error: ${err.message ?? err}`;
    setStatus(err.message ?? String(err), true);
  } finally {
    endGenerating();
    bubble.classList.remove("pending");
    // Not unconditionally true: stopping a Transformers.js/VQA generation
    // terminates its worker (see hardStopWorkerGeneration), which clears
    // loadedModelKey — re-deriving from it here (the same source of truth
    // clearChat()/loadSession() already use) correctly leaves the input
    // disabled in that case instead of re-enabling it for a dead engine.
    setChatEnabled(loadedModelKey === `${providerSelect.value}:${modelSelect.value}`);
    input.focus();
    // VQA sessions include an in-memory image and aren't persisted to
    // localStorage — see the note on `vqaImage`/`vqaMessages` above.
    if (!isVqa) upsertCurrentSession();
  }
});
