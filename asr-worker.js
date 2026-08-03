// Runs Whisper (automatic-speech-recognition) off the main thread, mirroring
// transformers-worker.js's rationale — a dedicated worker, not that one,
// because ASR's lifecycle is independent of whichever chat provider is
// currently loaded: switching chat providers must not re-download or reload
// the Whisper model, and loading Whisper must not disturb whatever chat
// engine is already loaded (see resetEngineHandles() in app.js, which this
// worker is deliberately never touched by).

let pipelineInstance = null;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      const { pipeline } = await import("https://esm.run/@huggingface/transformers");
      pipelineInstance = await pipeline("automatic-speech-recognition", msg.modelId, {
        dtype: msg.dtype,
        progress_callback: (progress) => {
          self.postMessage({ type: "progress", progress });
        },
      });
      self.postMessage({ type: "loaded" });
    } else if (msg.type === "transcribe") {
      // audioData is a Float32Array, already decoded+resampled to 16kHz on
      // the main thread (see decodeAudioBlob() in app.js) — NOT done here.
      // Passing a URL for this worker to fetch+decode itself was the first
      // approach tried; it failed with "AudioContext is not available in
      // your environment" (confirmed via direct testing) because Workers
      // don't have AudioContext, which the library's internal audio decoder
      // needs. A raw Blob doesn't work either (throws "expects ...
      // Float32Array ... got Blob"). Float32Array is what the pipeline
      // actually wants, and it's structured-cloneable through postMessage.
      const result = await pipelineInstance(msg.audioData);
      self.postMessage({ type: "transcript", text: result.text ?? "" });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message ?? String(err) });
  }
};
