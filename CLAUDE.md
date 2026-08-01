# Client-side LLM demo

Original goal:
- explore the various client-side (in-browser) JS LLM options available
- build a simple HTML demo with a tiny LLM (Chrome already embeds Gemini Nano)
- let the user chat with the client-side LLM

## What's here

A single-page, no-build-step chat demo — `index.html` + `app.js` + `style.css` — that runs an LLM entirely in the browser: no backend, no API key, no data leaving the machine. A dropdown switches between three interchangeable providers, all sharing one chat UI and message-history array.

## Providers and supported models

| Provider | Library | Runtime | Model | Notes |
| --- | --- | --- | --- | --- |
| WebLLM | `@mlc-ai/web-llm` | WebGPU | `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~1GB-ish quantized download, cached after first load |
| WebLLM | `@mlc-ai/web-llm` | WebGPU | `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | smaller/faster than the Llama option |
| Transformers.js | `@huggingface/transformers` | WebGPU, falls back to WASM | `onnx-community/Qwen2.5-0.5B-Instruct` (q4 quantized) | best quality-to-size tradeoff of the Transformers.js options |
| Transformers.js | `@huggingface/transformers` | WebGPU, falls back to WASM | `HuggingFaceTB/SmolLM2-135M-Instruct` | smallest/fastest; runs fine on WASM alone, no WebGPU required — best first thing to try |
| Chrome built-in AI | Prompt API (`LanguageModel`) | ships inside Chrome | Gemini Nano | zero download; fixed model, not user-selectable; needs `chrome://flags/#prompt-api-for-gemini-nano` or an origin trial |

Other client-side options that exist but weren't built into this demo (documented in `README.md`): wllama (llama.cpp compiled to WASM, runs GGUF models).

## Architecture

- `index.html` — page shell: provider selector, model selector (repopulated per-provider via JS, since WebLLM's MLC model IDs and Transformers.js's ONNX repo IDs are different namespaces), chat log, message input form.
- `app.js` — plain ES module, no bundler, all libraries loaded via CDN (`esm.run`) dynamic `import()`.
  - `providerModels` config maps each provider to its model list.
  - `loadWebLLM` / `loadTransformers` / `loadChromeAI` each populate one of three module-level engine handles (`webllmEngine`, `transformersPipeline`, `chromeSession`) — exactly one is live at a time.
  - `streamWebLLM` / `streamTransformers` / `streamChromeAI` stream tokens into the current assistant chat bubble as they arrive.
  - Shared `history` array of `{role, content}` messages feeds whichever provider is active.
- `style.css` — full-viewport flex layout (header/controls/status fixed height, chat log fills remaining space), responsive down to mobile widths.

## Known issue: Hugging Face CDN flakiness

Both WebLLM and Transformers.js pull model weights from `huggingface.co` on first load. That CDN intermittently serves a CloudFront error page instead of the real file; since error responses don't carry CORS headers, the browser reports it as a CORS failure rather than the real transient status — confirmed via direct `curl` testing (same URL flipping between 200 and 404 across successive requests) and via Chrome DevTools Protocol network inspection (`corsErrorStatus.corsError: "MissingAllowOriginHeader"` on what was actually a 404 CloudFront error page). This is not a bug in the app's CORS setup or Cloudflare config.

`withRetries()` in `app.js` retries model loads up to 3 times with backoff to smooth over this. If it still fails after that, the status line says so explicitly and suggests clicking "Load model" again.

## Running it

WebGPU and ES module imports require serving over `http(s)://`, not opened as a `file://` URL:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a recent Chrome or Edge.
