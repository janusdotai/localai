# clientai

A tiny demo of chatting with an LLM — and captioning or asking questions about an image — that runs **entirely in the browser**: no server, no API key, no data leaving your machine.

It's a single-page, no-build-step app (`index.html` + `app.js` + `style.css`) that switches between several client-side inference options — [WebLLM](https://github.com/mlc-ai/web-llm), [Transformers.js](https://github.com/huggingface/transformers.js), and Chrome's built-in Gemini Nano — with a familiar ChatGPT/OpenWebUI-style UI, persisted chat history, and a storage inspector for the downloaded model weights.

# Demo

https://local.janus.ai

## Running it

WebGPU and ES module imports require the page to be served over `http(s)://`, not opened directly as a `file://` URL:

```sh
python3 -m http.server 8000
# or: npx serve .
```

Then open `http://localhost:8000` in a recent Chrome or Edge, pick a provider from the dropdown, click "Load model", and start chatting (or upload an image, for the captioning/image-Q&A providers).

See [CLAUDE.md](CLAUDE.md) for full architecture notes, provider/model details, and known issues.
