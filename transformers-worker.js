// Runs Transformers.js off the main thread. WASM inference is synchronous —
// running it on the main thread blocks all UI updates and repaints for as
// long as generation takes, which is exactly what makes Chrome show the
// "Page Unresponsive" dialog on longer replies. A worker's event loop is
// separate from the page's, so the UI stays interactive no matter how long
// a generation takes.

let pipelineInstance = null;

// The VQA (image-text-to-text) model doesn't work through the high-level
// pipeline() API in the current @huggingface/transformers release — its
// pipeline registry doesn't include an "image-text-to-text" task at all
// (confirmed directly against the bundled source, not assumed). SmolVLM
// needs the lower-level AutoProcessor + AutoModelForVision2Seq API instead,
// with a manually driven generate() call.
let vqaProcessor = null;
let vqaModel = null;

// A "stop generating" button was tried here first via generate()'s
// stopping_criteria option (InterruptableStoppingCriteria) triggered by a
// postMessage from the main thread — confirmed via direct testing that the
// interrupt mechanism itself works instantly once it runs, but the postMessage
// carrying it sat unprocessed for ~80 seconds (out of ~90s total) during a
// real generation: this worker's event loop never yields back to the message
// queue during the WASM generation loop until it's nearly done anyway, making
// it useless for its actual purpose. Stopping generation on this provider is
// instead done from app.js via Worker.terminate() — a lower-level primitive
// that doesn't need the worker's cooperation at all.

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      const { pipeline } = await import("https://esm.run/@huggingface/transformers");
      pipelineInstance = await pipeline(msg.task, msg.modelId, {
        ...(msg.dtype ? { dtype: msg.dtype } : {}),
        progress_callback: (progress) => {
          self.postMessage({ type: "progress", progress });
        },
      });
      self.postMessage({ type: "loaded" });
    } else if (msg.type === "load-vqa") {
      const { AutoProcessor, AutoModelForVision2Seq } = await import(
        "https://esm.run/@huggingface/transformers"
      );
      const progress_callback = (progress) => {
        self.postMessage({ type: "progress", progress });
      };
      vqaProcessor = await AutoProcessor.from_pretrained(msg.modelId, { progress_callback });
      vqaModel = await AutoModelForVision2Seq.from_pretrained(msg.modelId, {
        ...(msg.dtype ? { dtype: msg.dtype } : {}),
        ...(msg.device ? { device: msg.device } : {}),
        progress_callback,
      });
      self.postMessage({ type: "loaded" });
    } else if (msg.type === "generate") {
      const { TextStreamer } = await import("https://esm.run/@huggingface/transformers");
      const streamer = new TextStreamer(pipelineInstance.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
          self.postMessage({ type: "token", text });
        },
      });
      await pipelineInstance(msg.history, {
        max_new_tokens: 2048,
        do_sample: false,
        streamer,
      });
      self.postMessage({ type: "done" });
    } else if (msg.type === "vqa-generate") {
      const { TextStreamer, RawImage } = await import("https://esm.run/@huggingface/transformers");

      // Each turn's content is either a plain string (text-only follow-up)
      // or an array of {type, text|image} parts (the first turn, which
      // carries the attached image). Collect the actual RawImage objects
      // separately — apply_chat_template only wants placeholders in the
      // message content, the real image data is passed to the processor call.
      const images = [];
      const templateMessages = [];
      for (const m of msg.messages) {
        if (Array.isArray(m.content)) {
          const parts = [];
          for (const part of m.content) {
            if (part.type === "image") {
              images.push(await RawImage.fromBlob(part.image));
              parts.push({ type: "image" });
            } else {
              parts.push(part);
            }
          }
          templateMessages.push({ role: m.role, content: parts });
        } else {
          // SmolVLM's chat template does `for item in message.content`, so
          // even text-only turns need content as a list of typed parts —
          // a raw string throws "Expected iterable or object type in for
          // loop: got StringValue" (confirmed via testing, not assumed).
          templateMessages.push({ role: m.role, content: [{ type: "text", text: m.content }] });
        }
      }

      const text = vqaProcessor.apply_chat_template(templateMessages, {
        add_generation_prompt: true,
      });
      const inputs = await vqaProcessor(text, images);

      const streamer = new TextStreamer(vqaProcessor.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk) => {
          self.postMessage({ type: "token", text: chunk });
        },
      });

      // Recomputing from the full conversation each turn (no past_key_values
      // reuse) rather than managing cross-call KV-cache state in the worker
      // — simpler, and fine at this model size, same tradeoff every other
      // provider in this app already makes by resending full history.
      await vqaModel.generate({
        ...inputs,
        do_sample: false,
        max_new_tokens: 2048,
        streamer,
      });
      self.postMessage({ type: "done" });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message ?? String(err) });
  }
};
