// Client-side LLM chat demo. Three providers, all running fully in-browser:
//   - "webllm": open-weight tiny models via WebGPU (@mlc-ai/web-llm)
//   - "transformers": open-weight tiny ONNX models via WASM/WebGPU (@huggingface/transformers)
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
  chrome: [],
};

const providerSelect = document.getElementById("provider");
const modelSelect = document.getElementById("model");
const modelGroup = document.getElementById("model-group");
const loadBtn = document.getElementById("load-btn");
const statusEl = document.getElementById("status");
const chatEl = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

/** @type {{role: "user"|"assistant", content: string}[]} */
let history = [];

// Active provider handle. Exactly one of these is set after a successful load.
let webllmEngine = null;
let transformersPipeline = null;
let chromeSession = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function setChatEnabled(enabled) {
  input.disabled = !enabled;
  sendBtn.disabled = !enabled;
}

function addBubble(role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
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

function resetProviderState() {
  webllmEngine = null;
  transformersPipeline = null;
  chromeSession = null;
  history = [];
  chatEl.innerHTML = "";
  setChatEnabled(false);
}

providerSelect.addEventListener("change", () => {
  populateModelOptions(providerSelect.value);
  resetProviderState();
  setStatus("");
});

populateModelOptions(providerSelect.value);

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
      setStatus(`Download hiccup, retrying (${attempt}/${maxAttempts})...`);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  }
}

async function loadWebLLM() {
  setStatus("Loading WebLLM engine...");
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
        setStatus(report.text || "Loading model...");
      },
    })
  );
  setStatus(`Loaded ${modelId}. Ready to chat.`);
}

async function loadTransformers() {
  setStatus("Loading Transformers.js...");
  const { pipeline } = await import(
    "https://esm.run/@huggingface/transformers"
  );

  const modelId = modelSelect.value;
  transformersPipeline = await withRetries(() =>
    pipeline("text-generation", modelId, {
      progress_callback: (progress) => {
        if (progress.status === "progress") {
          const pct = Math.round(progress.progress ?? 0);
          setStatus(`Downloading ${progress.file} (${pct}%)...`);
        } else {
          setStatus(`${progress.status}...`);
        }
      },
    })
  );
  setStatus(`Loaded ${modelId}. Ready to chat.`);
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

  setStatus("Preparing Chrome built-in model...");
  chromeSession = await LanguageModel.create({
    monitor(m) {
      m.addEventListener("downloadprogress", (e) => {
        setStatus(`Downloading Gemini Nano: ${Math.round(e.loaded * 100)}%`);
      });
    },
  });
  setStatus("Gemini Nano ready. Ready to chat.");
}

const loaders = {
  webllm: loadWebLLM,
  transformers: loadTransformers,
  chrome: loadChromeAI,
};

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  setChatEnabled(false);
  resetProviderState();
  try {
    await loaders[providerSelect.value]();
    setChatEnabled(true);
  } catch (err) {
    console.error(err);
    setStatus(err.message ?? String(err), true);
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
