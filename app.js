// Client-side LLM chat demo. Four providers, all running fully in-browser:
//   - "webllm": open-weight tiny chat models via WebGPU (@mlc-ai/web-llm)
//   - "transformers": open-weight tiny chat models via WASM/WebGPU (@huggingface/transformers)
//   - "caption": image captioning via WASM/WebGPU (@huggingface/transformers)
//   - "chrome": Chrome's built-in Gemini Nano (window.LanguageModel Prompt API)

const providerModels = {
  webllm: [
    { value: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B Instruct (q4f16)" },
    { value: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 0.5B Instruct (q4f16)" },
  ],
  transformers: [
    { value: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen 2.5 0.5B Instruct (ONNX, q4)", dtype: "q4" },
    { value: "HuggingFaceTB/SmolLM2-135M-Instruct", label: "SmolLM2 135M Instruct (ONNX, very small/fast)" },
  ],
  caption: [
    // The default quantized export of this model has a broken decoder graph
    // ("Missing required scale" DequantizeLinear error) under current ONNX
    // Runtime Web — fp32 sidesteps it at the cost of a bigger download.
    { value: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Image Captioning", dtype: "fp32" },
  ],
  chrome: [],
};

const providerSelect = document.getElementById("provider");
const modelSelect = document.getElementById("model");
const modelGroup = document.getElementById("model-group");
const loadBtn = document.getElementById("load-btn");
const statusEl = document.getElementById("status");
const chatEl = document.getElementById("chat");
const form = document.getElementById("chat-form");
const textControls = document.getElementById("text-controls");
const imageControls = document.getElementById("image-controls");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const imageInput = document.getElementById("image-input");
const imageLabel = document.getElementById("image-label");
const loadModal = document.getElementById("load-modal");
const modalModelName = document.getElementById("modal-model-name");
const modalProgressFill = document.getElementById("modal-progress-fill");
const modalStatus = document.getElementById("modal-status");
const modalCloseBtn = document.getElementById("modal-close-btn");

/** @type {{role: "user"|"assistant", content: string}[]} */
let history = [];

// Active provider handle. Exactly one of these is set after a successful load.
let webllmEngine = null;
let transformersPipeline = null;
let captionerPipeline = null;
let chromeSession = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// Downloading a model is a real commitment — tens to hundreds of MB over the
// network — so make it obvious with a blocking full-page modal and progress
// bar rather than a quiet status line, and let the library-reported progress
// (0-100, or unknown/indeterminate) drive the bar.
function showLoadModal(modelId) {
  modalModelName.textContent = modelId;
  modalProgressFill.classList.add("indeterminate");
  modalProgressFill.style.width = "0%";
  modalStatus.textContent = "Starting…";
  modalStatus.classList.remove("error");
  modalCloseBtn.classList.add("hidden");
  loadModal.classList.remove("hidden");
}

function hideLoadModal() {
  loadModal.classList.add("hidden");
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

modalCloseBtn.addEventListener("click", hideLoadModal);

function setChatEnabled(enabled) {
  input.disabled = !enabled;
  sendBtn.disabled = !enabled;
  imageInput.disabled = !enabled;
  imageLabel.classList.toggle("disabled", !enabled);
}

function addBubble(role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
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
}

function resetProviderState() {
  webllmEngine = null;
  transformersPipeline = null;
  captionerPipeline = null;
  chromeSession = null;
  history = [];
  chatEl.innerHTML = "";
  imageInput.value = "";
  setChatEnabled(false);
}

providerSelect.addEventListener("change", () => {
  populateModelOptions(providerSelect.value);
  syncInputMode(providerSelect.value);
  resetProviderState();
  setStatus("");
});

populateModelOptions(providerSelect.value);
syncInputMode(providerSelect.value);

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

async function loadTransformers() {
  setLoadProgress("Loading Transformers.js...");
  const { pipeline } = await import(
    "https://esm.run/@huggingface/transformers"
  );

  const modelId = modelSelect.value;
  const { dtype } = getSelectedModelConfig();
  transformersPipeline = await withRetries(() =>
    pipeline("text-generation", modelId, {
      ...(dtype ? { dtype } : {}),
      progress_callback: transformersProgressCallback,
    })
  );
  setLoadProgress(`Loaded ${modelId}. Ready to chat.`, 100);
}

async function loadCaptioner() {
  setLoadProgress("Loading image captioning model...");
  const { pipeline } = await import(
    "https://esm.run/@huggingface/transformers"
  );

  const modelId = modelSelect.value;
  const { dtype } = getSelectedModelConfig();
  captionerPipeline = await withRetries(() =>
    pipeline("image-to-text", modelId, {
      ...(dtype ? { dtype } : {}),
      progress_callback: transformersProgressCallback,
    })
  );
  setLoadProgress(`Loaded ${modelId}. Upload an image to caption it.`, 100);
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
  chrome: loadChromeAI,
};

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  setChatEnabled(false);
  resetProviderState();
  showLoadModal(getSelectedModelConfig()?.label ?? providerSelect.value);
  try {
    await loaders[providerSelect.value]();
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

async function streamWebLLM(bubble) {
  const chunks = await webllmEngine.chat.completions.create({
    messages: history,
    stream: true,
  });
  let full = "";
  for await (const chunk of chunks) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    full += delta;
    bubble.textContent = full;
    chatEl.scrollTop = chatEl.scrollHeight;
  }
  return full;
}

async function streamTransformers(bubble) {
  const { TextStreamer } = await import(
    "https://esm.run/@huggingface/transformers"
  );

  let full = "";
  const streamer = new TextStreamer(transformersPipeline.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      full += text;
      bubble.textContent = full;
      chatEl.scrollTop = chatEl.scrollHeight;
    },
  });

  await transformersPipeline(history, {
    max_new_tokens: 512,
    do_sample: false,
    streamer,
  });
  return full;
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
    bubble.textContent = full;
    chatEl.scrollTop = chatEl.scrollHeight;
  }
  return full;
}

const streamers = {
  webllm: streamWebLLM,
  transformers: streamTransformers,
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
    const output = await captionerPipeline(file);
    bubble.textContent = output[0]?.generated_text ?? "(no caption produced)";
  } catch (err) {
    console.error(err);
    bubble.textContent = `Error: ${err.message ?? err}`;
    setStatus(err.message ?? String(err), true);
  } finally {
    bubble.classList.remove("pending");
    setChatEnabled(true);
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  setChatEnabled(false);

  history.push({ role: "user", content: text });
  addBubble("user").textContent = text;

  const bubble = addBubble("assistant");
  bubble.classList.add("pending");

  try {
    const reply = await streamers[providerSelect.value](bubble);
    history.push({ role: "assistant", content: reply });
  } catch (err) {
    console.error(err);
    bubble.textContent = `Error: ${err.message ?? err}`;
    setStatus(err.message ?? String(err), true);
  } finally {
    bubble.classList.remove("pending");
    setChatEnabled(true);
    input.focus();
  }
});
