# clientai

A tiny demo of chatting with an LLM that runs **entirely in the browser** — no server, no API key, no data leaving your machine.

Open `index.html` (via a static server, see below), pick a provider, load the model, and chat.

## The client-side LLM landscape

| Approach | How it runs | Tradeoffs |
| --- | --- | --- |
| **WebLLM** (`@mlc-ai/web-llm`) — used here | Open-weight quantized models (Llama 3.2 1B, Qwen 2.5 0.5B, etc.) compiled to run via **WebGPU**, streamed from a CDN and cached by the browser after first load | Real, swappable models; needs a WebGPU browser (Chrome/Edge, and increasingly others) and a several-hundred-MB download on first use |
| **Chrome built-in AI (Gemini Nano)** — used here | Ships inside Chrome itself, exposed via the origin-trial **Prompt API** (`LanguageModel.create()`) | Zero download, fastest to start; Chrome-only, requires enabling `chrome://flags/#prompt-api-for-gemini-nano` (or an origin trial token), and the model itself isn't user-chosen |
| Transformers.js / ONNX Runtime Web | Small models run via WASM (optionally WebGPU) through Hugging Face's JS runtime | Broadest browser compatibility (no WebGPU required), but slower and limited to smaller/older architectures |
| wllama | llama.cpp compiled to WASM, runs GGUF models | Good for llama.cpp-family models client-side; WASM-only means slower than WebGPU-backed WebLLM |

This demo implements the first two, since together they cover both ends of the tradeoff (bring-your-own tiny model vs. zero-download built-in model) with no build step required.

## Running it

WebGPU and ES module imports require the page to be served over `http(s)://`, not opened directly as a `file://` URL. From this directory:

```sh
python3 -m http.server 8000
# or: npx serve .
```

Then open `http://localhost:8000` in a recent Chrome or Edge.

- **WebLLM**: select it, pick a model, click "Load model". First load downloads and compiles the model (progress shown), then it's cached for next time.
- **Chrome built-in Gemini Nano**: select it, click "Load model". If unavailable, the status line explains why (usually the `chrome://flags/#prompt-api-for-gemini-nano` flag needs enabling).
