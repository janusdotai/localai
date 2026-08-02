# clientai

A tiny demo of chatting with an LLM — and captioning an image — that runs **entirely in the browser**: no server, no API key, no data leaving your machine.

Open `index.html` (via a static server, see below), pick a provider, load the model, and chat (or upload an image, for the captioning provider).

The UI follows the familiar ChatGPT/OpenWebUI layout: a collapsible left sidebar (hamburger icon, or the "+ New chat"/"+" buttons, toggle it) with a real, persisted chat history. Every text conversation (WebLLM, Transformers.js chat, or Chrome's Gemini Nano) is saved to your browser's `localStorage` as you go, titled from your first message, and listed in the sidebar — click one to reload it (you may need to click "Load model" again first if you've since switched models, but your conversation will still be there once it's loaded). Hover a chat for a **×** to delete it, or use **Clear all** above the list to wipe everything. "New chat" clears the visible conversation but keeps whatever model is already loaded — no re-download needed. Image captioning isn't a multi-turn conversation, so it isn't saved to history.

## The client-side LLM landscape

| Approach | How it runs | Tradeoffs |
| --- | --- | --- |
| **WebLLM** (`@mlc-ai/web-llm`) — used here | Open-weight quantized models (Llama 3.2 1B, Qwen 2.5 0.5B, etc.) compiled to run via **WebGPU**, streamed from a CDN and cached by the browser after first load | Real, swappable models; needs a WebGPU browser (Chrome/Edge, and increasingly others) and a several-hundred-MB download on first use |
| **Transformers.js** (`@huggingface/transformers`) — used here | Open-weight models run as **ONNX** via ONNX Runtime — WebGPU when available, WASM (pure CPU) otherwise | Broadest browser compatibility (works even without WebGPU); ONNX Runtime doesn't ship model weights with the CDN bundle, so first load still fetches from Hugging Face — WASM inference is noticeably slower than WebGPU |
| **Chrome built-in AI (Gemini Nano)** — used here | Ships inside Chrome itself, exposed via the origin-trial **Prompt API** (`LanguageModel.create()`) | Zero download, fastest to start; Chrome-only, requires enabling `chrome://flags/#prompt-api-for-gemini-nano` (or an origin trial token), and the model itself isn't user-chosen |
| wllama | llama.cpp compiled to WASM, runs GGUF models | Good for llama.cpp-family models client-side; WASM-only means slower than WebGPU-backed WebLLM |

This demo implements the first three for text chat, plus a fourth mode using Transformers.js for **image captioning** (a vision task, not text chat) — covering both ends of the text tradeoff curve (bring-your-own model over WebGPU vs. WASM vs. zero-download built-in) and showing the same runtime handles more than just chat, with no build step required.

## Running it

WebGPU and ES module imports require the page to be served over `http(s)://`, not opened directly as a `file://` URL. From this directory:

```sh
python3 -m http.server 8000
# or: npx serve .
```

Then open `http://localhost:8000` in a recent Chrome or Edge.

- **WebLLM**: select it, pick a model, click "Load model". First load downloads and compiles the model (progress shown), then it's cached for next time.
- **Transformers.js**: select it, pick a model (SmolLM2 135M is fastest to try first — no WebGPU required), click "Load model". Also cached after first load.
- **Transformers.js (image captioning)**: select it, click "Load model", then use "Choose an image…" to upload a photo — it's captioned on-device (no chat, just image in, caption out).
- **Chrome built-in Gemini Nano**: select it, click "Load model". If unavailable, the status line explains why (usually the `chrome://flags/#prompt-api-for-gemini-nano` flag needs enabling).

"Load model" always asks for confirmation first, naming the model and its approximate download size, before anything downloads. Click the **Storage** button in the sidebar any time to see which models are actually cached in your browser and how much space each one is using — models are downloaded via the browser's Cache API (not `localStorage`, not tied to your tab session), so they persist across refreshes and restarts until you clear them from that panel or clear the site's data in your browser.

WebLLM and both Transformers.js modes download model weights from Hugging Face on first use; that CDN occasionally serves a transient error for a request (surfaces in the browser as a CORS-looking failure). The app retries automatically a few times before giving up — if it still fails, just click "Load model" again.

If you deploy this to a `*.workers.dev` subdomain: Hugging Face's CDN blocks requests whose `Referer` is a `workers.dev` domain (likely anti-scraping protection), which made model downloads fail every time on that hosting. `index.html` already includes `<meta name="referrer" content="no-referrer">` to work around it — no action needed, but worth knowing if you fork this and see it fail consistently only when deployed (and not locally).
