// Dedicated GPU worker. Loads the WASM module (same binary the prover thread
// uses) and runs the async `runGpuWorker(sab)` entry, which loops forever
// processing commands from the SAB and dispatching to wgpu.
//
// The bootstrap on the prover side (gpu-bootstrap.js) creates the
// SharedArrayBuffer, spawns this worker as a module Worker, posts an INIT
// message, and awaits a READY response.
//
// Mirrors the wasm import pattern used by `web-client-methods-worker.js` —
// the rollup `rewriteWorkerWasmImport` plugin rewrites `../../dist/wasm.js`
// to the variant-specific path at build time.

import loadWasm from "../../dist/wasm.js";

let initPromise = null;

self.onmessage = async (event) => {
  if (!event.data || event.data.type !== "INIT") return;
  if (initPromise) return; // ignore duplicate INIT

  initPromise = (async () => {
    try {
      const wasm = await loadWasm();
      if (!wasm || typeof wasm.runGpuWorker !== "function") {
        throw new Error(
          "WASM module loaded but `runGpuWorker` export is missing — was the SDK built with `--features real-gpu`?"
        );
      }
      // Kick off the async command loop. It runs forever; we don't await.
      // If runGpuWorker rejects (e.g. GPU init fails), surface the error.
      wasm.runGpuWorker(event.data.sab).catch((err) => {
        self.postMessage({
          type: "ERROR",
          error: String((err && err.message) || err),
        });
      });
      self.postMessage({ type: "READY" });
    } catch (err) {
      self.postMessage({
        type: "ERROR",
        error: String((err && err.message) || err),
      });
    }
  })();
};
