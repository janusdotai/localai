// Runs Transformers.js off the main thread. WASM inference is synchronous —
// running it on the main thread blocks all UI updates and repaints for as
// long as generation takes, which is exactly what makes Chrome show the
// "Page Unresponsive" dialog on longer replies. A worker's event loop is
// separate from the page's, so the UI stays interactive no matter how long
// a generation takes.

let pipelineInstance = null;

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
        max_new_tokens: 512,
        do_sample: false,
        streamer,
      });
      self.postMessage({ type: "done" });
    } else if (msg.type === "caption") {
      const output = await pipelineInstance(msg.file);
      self.postMessage({ type: "caption-result", text: output[0]?.generated_text ?? "" });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message ?? String(err) });
  }
};
