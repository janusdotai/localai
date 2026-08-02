# Client-side LLM demo

Original goal:
- explore the various client-side (in-browser) JS LLM options available
- build a simple HTML demo with a tiny LLM (Chrome already embeds Gemini Nano)
- let the user chat with the client-side LLM

## What's here

A single-page, no-build-step demo — `index.html` + `app.js` + `style.css` (+ `transformers-worker.js`, see below) — that runs models entirely in the browser: no backend, no API key, no data leaving the machine. A dropdown switches between four interchangeable providers/modes; three are text chat sharing one chat UI and message-history array, the fourth swaps the text input for an image upload control and does captioning instead.

## Providers and supported models

| Provider | Library | Runtime | Model | Notes |
| --- | --- | --- | --- | --- |
| WebLLM | `@mlc-ai/web-llm` | WebGPU | `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~1GB-ish quantized download, cached after first load |
| WebLLM | `@mlc-ai/web-llm` | WebGPU | `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | smaller/faster than the Llama option |
| Transformers.js (chat) | `@huggingface/transformers` | WebGPU, falls back to WASM | `onnx-community/Qwen2.5-0.5B-Instruct` (`dtype: "q4"`) | best quality-to-size tradeoff of the Transformers.js chat options |
| Transformers.js (chat) | `@huggingface/transformers` | WebGPU, falls back to WASM | `HuggingFaceTB/SmolLM2-135M-Instruct` | smallest/fastest; runs fine on WASM alone, no WebGPU required — best first thing to try |
| Transformers.js (image captioning) | `@huggingface/transformers` | WebGPU, falls back to WASM | `Xenova/vit-gpt2-image-captioning` (`dtype: "fp32"`) | canonical, widely-used ViT-encoder/GPT2-decoder captioning model built for this library; verified end-to-end (upload → thumbnail bubble → caption) |
| Chrome built-in AI | Prompt API (`LanguageModel`) | ships inside Chrome | Gemini Nano | zero download; fixed model, not user-selectable; needs `chrome://flags/#prompt-api-for-gemini-nano` or an origin trial |

Other client-side options that exist but weren't built into this demo (documented in `README.md`): wllama (llama.cpp compiled to WASM, runs GGUF models).

### Known issue: this model's default quantized export is broken

`Xenova/vit-gpt2-image-captioning`'s default quantized ONNX graph fails session creation under the current ONNX Runtime Web (`Missing required scale ... DequantizeLinear` on the decoder). Forcing `dtype: "fp32"` in its `providerModels` entry sidesteps the broken graph at the cost of a larger download (encoder + decoder in full precision). Confirmed via direct testing: quantized → deterministic ORT session-creation error on every attempt; fp32 → loads and captions correctly.

## Architecture

- `index.html` — an OpenWebUI/ChatGPT-style shell: `.app-shell` = collapsible `#sidebar` + `.main`. The sidebar holds a "+ New chat" button, a real, persisted chat-history list (`#sidebar-chats`, populated by `renderSidebarChats()` — see "Chat history persistence" below), and the real provider/model selectors + Load/Storage buttons (same ids as before, just relocated — see below). `.main` has a topbar (hamburger sidebar-toggle + current model name + "+" new-chat), the scrolling `#chat` (messages append into `#chat-inner`, a centered `max-width: 48rem` column — `#chat` itself just provides the scrollbar), and `.composer` (the existing `#chat-form` restyled into a rounded pill, unchanged `#text-controls`/`#image-controls` swap logic for chat vs. captioning).
- `app.js` — plain ES module, no bundler, all libraries loaded via CDN (`esm.run`) dynamic `import()`. The restyle deliberately reused every existing element id so almost none of the functional code needed to change — only relocated in the DOM. What's new:
  - `clearChat()` (wired to both "New chat" buttons) resets `history`/`chatInner` **without** touching the loaded engine — so, like ChatGPT, starting a new chat keeps whatever model is already loaded; `resetProviderState()` (used on provider/model switch, where the old engine really is invalid) now calls `clearChat()` plus nulls the engine handles.
  - `addBubble()` now builds a `.msg-row` (avatar + bubble) and appends into `chatInner` instead of `chatEl` directly — still returns the bubble node, so every existing `bubble.textContent = ...` call site elsewhere is untouched.
  - `toggleSidebar()`/`setSidebarOpen()` — width-collapse on desktop, slide-in overlay drawer with a backdrop under a 768px breakpoint; state persisted in `localStorage` (`sidebarOpen` — a legitimate use of it, unlike model weights, which never belong in localStorage).
  - `updateTopbarTitle()` reflects the current provider+model *selection* (not load state) in the topbar; fired on provider change, model change, and once at init.
  - `providerModels` config maps each provider to its model list; each entry can carry a `dtype` (read by `getSelectedModelConfig()` and passed to `pipeline()` for the two Transformers.js providers — needed both for download-size control and to work around the captioning model's broken default quant, below).
  - `loadWebLLM` / `loadTransformers` / `loadCaptioner` / `loadChromeAI` each populate one of four module-level engine handles (`webllmEngine`, `transformersPipeline`, `captionerPipeline`, `chromeSession`) — exactly one is live at a time.
  - `streamWebLLM` / `streamTransformers` / `streamChromeAI` stream tokens into the current assistant chat bubble as they arrive; the `imageInput` `change` handler runs the non-streaming captioning pipeline instead and renders the uploaded image via `addImageBubble()`.
- `style.css` — flex shell (`.app-shell` → `.sidebar` + `.main`) instead of the old single centered column; sidebar collapse/drawer breakpoint at 768px (separate from the existing 480px mobile breakpoint used for finer spacing tweaks).

### Icons and empty-state layout (matching real OpenWebUI/Grok screenshots)

Restyled a second time against actual OpenWebUI/Grok screenshots the user provided — swapped the earlier CSS-drawn hamburger and text-label buttons for a small inline-SVG icon set (Lucide-style thin-line icons, hand-written directly in the HTML/JS, no icon library/font — stays dependency-free): pencil for New Chat (sidebar nav item + topbar button), a panel-rectangle for the sidebar toggle, a database/cylinder icon for Storage, a trash icon for per-chat delete (was a bare `×`), an arrow-up in a circle for Send, and a small image icon on the "Choose an image…" upload button. Deliberately did **not** add icons/nav rows for things this app doesn't have (the reference's Search/Notes/Workspace items) — icons only went on buttons that already did something.

The other half of that match: before any messages exist, `.composer` becomes an absolutely-centered overlay in `.main` (a small chat-bubble icon + the current model name above the input, like the reference's centered hero) instead of sitting pinned to the bottom; `updateEmptyState()` toggles a `.main.is-empty` class based on whether `#chat-inner` has any children, called from `addBubble()` and `clearChat()`. No new DOM reparenting — `.chat` just stays empty behind the overlay until the first message lands, at which point the composer reverts to normal bottom-pinned flow.

### Download UX: confirm dialog, progress modal, storage inspector

Clicking "Load model" doesn't download anything by itself — it opens a confirm dialog (`showConfirmModal()`) naming the model and an approximate size (`sizeEstimate` in `providerModels`, hand-estimated, not measured) before anything touches the network. Only clicking "Download" (`modalConfirmBtn`) starts the actual load, swapping the same `#load-modal` into its progress view (title/section swap, no separate modal element) driven by `setLoadProgress(text, percent)` — `percent` is `null` for indeterminate phases (library hasn't reported real progress yet) or a 0-100 number once it has.

A second, independent modal (`#settings-modal`, opened via the "Storage" button in the header) is a storage inspector: it lists every model in `providerModels` (except Chrome's, which downloads outside this page's visibility) and, for each, scans `caches` (the Cache API — not `localStorage`/`sessionStorage`) for entries whose URL contains that model's id, reporting downloaded/not-downloaded and real size (`Content-Length` header, falling back to reading the blob if absent). `navigator.storage.estimate()` supplies the overall usage/quota line. Per-model and bulk "Clear" buttons call `cache.delete()` on just the matching entries — confirmed via CDP testing that a fresh profile shows everything as "Not downloaded", downloading a model makes it show up with an accurate size, and Clear removes it again.

Both modals share the `.modal-overlay`/`.modal-card` styling and the same dismiss pattern: backdrop click and Escape close them when in a cancelable state (`isModalCancelable()` for the load modal — confirm step or post-error, never mid-download).

### Chat history persistence

Saved conversations live in `localStorage` under the key `chatSessions` (a JSON array of `{id, title, provider, modelValue, history, createdAt, updatedAt}`) — plain text, a few KB per conversation, so unlike model weights this is exactly what localStorage is for (no Cache API needed here). Only the three text-chat providers persist; image captioning is a one-shot action rather than a conversation (it never touches `history` at all), so it's intentionally excluded.

- `upsertCurrentSession()` runs in the `finally` block of the chat form's submit handler (so the user's message is saved even if the model errored), creating a new session on the first exchange (`currentSessionId` was null) or updating the existing one otherwise. Title is derived from the first user message (`deriveTitle()`, truncated ~42 chars), fixed at creation like ChatGPT's.
- `clearChat()` — called by both "New chat" buttons and by `resetProviderState()` on provider/model switch — sets `currentSessionId = null`, so the *next* exchange starts a new sidebar entry rather than overwriting whichever session was previously active.
- `loadSession(id)` restores a saved conversation for viewing: sets `history`, replays each message as a bubble, and switches `providerSelect`/`modelSelect` to match **without** dispatching their `change` events (so `resetProviderState()` doesn't fire and wipe what was just restored). Whether the input is actually usable depends on `loadedModelKey` (`"provider:modelValue"` of whatever engine is *actually* loaded right now, set on successful load, distinct from the dropdowns' current selection) — if it matches, chat is immediately usable; if not, input stays disabled with a "Load this model to continue the conversation" hint until the user clicks Load.
  - **Bug fixed during implementation**: the "Load model" confirm-modal handler used to call `resetProviderState()`, which also clears the chat display — so clicking Load to continue a just-restored session would immediately wipe the very history the user was trying to continue. Fixed by splitting `resetProviderState()` into `resetEngineHandles()` (nulls the engine handles + `loadedModelKey` only) and `resetProviderState()` (that, plus `clearChat()`) — the confirm-modal handler now calls only the former. Verified via CDP: restore a session, click Load, confirm the conversation is still on screen after the load completes.
- `deleteSession(id)` / `clearAllSessions()` (wired to each row's `.sidebar-chat-delete` × button, hover-revealed, and the "Clear all" link next to the Chats section header) remove from the array and re-save; deleting the currently-active session also clears the visible chat.
- All of the above verified end-to-end via CDP: two sessions created and correctly titled/sorted, switching between them restores the right content, delete removes the right one, sessions survive a full page reload, and reload → click a restored session → click Load model → history still present after load.

## Known issue: Hugging Face blocks `*.workers.dev` referrers (fixed)

When deployed to a Cloudflare Workers `*.workers.dev` subdomain, every model download from `huggingface.co` failed deterministically with what Chrome reported as a CORS error (`corsErrorStatus.corsError: "MissingAllowOriginHeader"`), even though the server's CORS headers were correct. Root-caused via a curl header-by-header bisection against the live deployed URL: Hugging Face's CDN returns a CloudFront 404 error page — with no CORS headers, hence the browser's misleading CORS-shaped error — specifically when the request's `Referer` header is any `*.workers.dev` domain (tested two unrelated `workers.dev` subdomains, both blocked; `pages.dev`/`example.com`/`localhost` referrers all succeeded). This is almost certainly anti-scraping protection against that shared, anonymous subdomain space, not a bug in this app's CORS/fetch setup.

Fix: `<meta name="referrer" content="no-referrer">` in `index.html` suppresses the `Referer` header on all outgoing requests from the page, including the ones WebLLM/Transformers.js issue internally — confirmed via CDP that no request carries a `Referer` header after adding it. This also explains why WebLLM appeared to fail even after the retry logic below was added: a real browser always sends `Referer` by default, so every attempt was hitting the same deterministic block; earlier `curl` diagnostics happened not to set a `Referer` and so only ever observed the separate, genuinely intermittent CDN flakiness described next.

## Known issue: Hugging Face CDN flakiness (separate from the above)

Independent of the referrer block, `huggingface.co` intermittently serves a CloudFront error page instead of the real file on a plain, unblocked request (confirmed via repeated `curl` with no special headers: same URL flipping between 200 and 404 across successive requests). `withRetries()` in `app.js` retries model loads up to 3 times with backoff to smooth over this. If it still fails after that, the status line says so explicitly and suggests clicking "Load model" again.

## Fixed: Transformers.js could freeze the tab ("Page Unresponsive")

Root cause: `@huggingface/transformers`'s WASM backend runs inference synchronously on whatever thread calls it. Running it on the main thread (the original implementation) meant a long generation blocked all UI updates/repaints for the entire duration — Chrome's own "Page Unresponsive" dialog is just the browser's generic response to a main thread that stops processing messages, and a multi-hundred-token WASM generation was easily long enough to trigger it. WebLLM (WebGPU) wasn't affected the same way since the heavy lifting happens on the GPU asynchronously; this was specific to the Transformers.js provider (both chat and image captioning, since both use the same WASM-backed library).

Fix: `transformers-worker.js` is a dedicated Web Worker that owns the actual `pipeline()` instance and does all loading/generation/captioning — the main thread only ever sends small messages (`{type: "load"|"generate"|"caption", ...}`) and receives progress/token/result messages back. `app.js`'s `transformersPipeline`/`captionerPipeline` are now just booleans (the real objects never leave the worker); `workerLoad()`/`workerGenerate()`/`workerCaption()` wrap the postMessage round-trip in a Promise so the rest of the app (`loadTransformers`, `loadCaptioner`, `streamTransformers`, the image-upload handler) barely changed. `resetEngineHandles()` terminates the worker on provider/model switch so a stale model isn't left resident in memory.

## Markdown rendering (assistant messages only)

Assistant replies render as markdown via [marked](https://github.com/markedjs/marked) (loaded lazily via the same `esm.run` CDN pattern as everything else — `getMarked()` in `app.js`, cached after first load). `renderAssistantMarkdown(bubble, text)` is called from all three streaming functions (`streamWebLLM`, `streamTransformers`, `streamChromeAI` — re-parses the full accumulated text on every token, which is cheap at this scale) and from `loadSession()` when replaying a saved conversation's assistant turns. User messages are untouched — always plain `bubble.textContent =`, never markdown-rendered — so nothing a user types can ever execute as HTML.

`marked.parse()` does not sanitize its output (raw HTML in the source would pass straight through), which normally means pairing it with something like DOMPurify. Deliberately not doing that here: this app is 100% client-side with no server and no other users, so the realistic worst case of unsanitized model output is self-XSS in your own tab from your own locally-run model's own reply. Documented with a comment at `getMarked()`/`renderAssistantMarkdown()` in `app.js` — add a sanitizer before reusing this rendering pattern anywhere with shared/multi-user state. If the CDN fetch of marked fails, `renderAssistantMarkdown()` catches the error and falls back to plain `textContent` rather than breaking the chat.

`.msg.assistant` in `style.css` switches from `white-space: pre-wrap` (used for plain text) to `white-space: normal` plus explicit spacing rules for `p`/`ul`/`ol`/`pre`/`code`/`blockquote`/headings — needed because marked's HTML output has its own whitespace between tags that `pre-wrap` would otherwise also preserve, adding unwanted gaps. A `--code-bg` custom property (separate light/dark values) backs inline `code` and `pre` blocks.

Verified via CDP: asked a loaded model for a markdown list with bold text and confirmed the actual DOM contains real `<ol>`/`<li>`/`<strong>` elements (not literal asterisks), while the user's own message — which also contained `**bold**` markdown syntax — stayed literal text, confirming the user-message/assistant-message rendering paths are correctly separated.

Verified via CDP: while a generation was in flight, repeated trivial `Runtime.evaluate` round-trips against the page averaged ~1ms (max 4.5ms) across 76 samples — the main thread was never blocked, which is the direct, measurable fix for the freeze. Also re-verified the full provider-switch lifecycle (worker terminate → fresh worker on next load) and image captioning, both working through the worker.

## Known issue: large cached files can silently fail to persist

Observed via the storage inspector during regression testing after the sidebar restyle: a freshly-downloaded ~130MB SmolLM2 model showed as cached but only ~2MB in size, even though inference worked correctly (weights were fetched and used in-memory) and a second chat message worked without any re-download in the same session. Console showed the same non-fatal `Cache.put() encountered a network error` warning previously noted for the ONNX Runtime WASM binary (see Transformers.js provider row above) — this time apparently affecting the actual model weight file, not just the runtime binary. Net effect: the *session* is fine, but the *persisted* cache entry for large files may be smaller than expected, meaning a later visit could need to re-fetch more than the storage inspector's numbers would suggest. Not something this app's code controls (it's the browser's Cache API opaque-response handling for large cross-origin responses) and not reliably reproducible attempt-to-attempt, so not fixed — just documented here in case the storage inspector's numbers look inconsistent across sessions.

## Running it

WebGPU and ES module imports require serving over `http(s)://`, not opened as a `file://` URL:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a recent Chrome or Edge.
