// Spawns the GPU worker, allocates the SAB, and resolves with the SAB once
// the worker signals READY. Called from the wasm-bindgen `init_dft` path on
// wasm32 (see `crates/web-client/src/models/provers.rs::WebGpuTransactionProver::
// init_dft`). The returned SAB is then handed to `WebGpuDft::new_with_sab` on
// the Rust side.
//
// SharedArrayBuffer requires cross-origin isolation (COOP/COEP). The SDK's
// MT build already requires COI; the GPU build inherits.

// Compute the worker URL lazily inside the function so webpack/Next's
// static `new URL(..., import.meta.url)` scanner doesn't trip when this
// module is also pulled into the GPU worker bundle (which never calls
// bootstrap). The template-literal concatenation prevents the static
// scan; runtime resolution still works correctly from the main bundle's
// import.meta.url.
function _computeWorkerUrl() {
  const path = `./${"workers"}/web-gpu-worker.module.js`;
  return new URL(path, import.meta.url);
}

export async function bootstrapGpuWorker(sabSize) {
  const workerUrl = _computeWorkerUrl();
  if (typeof SharedArrayBuffer !== "function") {
    throw new Error(
      "SharedArrayBuffer is not available — page must be cross-origin isolated (COOP: same-origin, COEP: require-corp)."
    );
  }
  const sab = new SharedArrayBuffer(sabSize);
  const worker = new Worker(workerUrl, { type: "module" });

  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    worker.onmessage = (e) => {
      if (!e.data) return;
      if (e.data.type === "READY") {
        finish(resolve, sab);
      } else if (e.data.type === "ERROR") {
        worker.terminate();
        finish(reject, new Error(`gpu worker error: ${e.data.error}`));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      finish(reject, new Error(`gpu worker boot error: ${e.message || e}`));
    };
    worker.postMessage({ type: "INIT", sab });
  });
}
