// Client-side LLM chat demo. Two providers, both running fully in-browser:
//   - "webllm": open-weight tiny models via WebGPU (@mlc-ai/web-llm)
//   - "chrome": Chrome's built-in Gemini Nano (window.LanguageModel Prompt API)

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

function syncModelVisibility() {
  const isWebllm = providerSelect.value === "webllm";
  modelGroup.style.display = isWebllm ? "" : "none";
}

function resetProviderState() {
  webllmEngine = null;
  chromeSession = null;
  history = [];
  chatEl.innerHTML = "";
  setChatEnabled(false);
}

providerSelect.addEventListener("change", () => {
  syncModelVisibility();
  resetProviderState();
  setStatus("");
});

syncModelVisibility();

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
  webllmEngine = await CreateMLCEngine(modelId, {
    initProgressCallback: (report) => {
      setStatus(report.text || "Loading model...");
    },
  });
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

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  setChatEnabled(false);
  resetProviderState();
  try {
    if (providerSelect.value === "webllm") {
      await loadWebLLM();
    } else {
      await loadChromeAI();
    }
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
    const reply =
      providerSelect.value === "webllm"
        ? await streamWebLLM(bubble)
        : await streamChromeAI(bubble);
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
