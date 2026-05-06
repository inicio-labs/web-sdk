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
  // Honor a host-set override first. Next.js dev with `file:` package deps
  // resolves `import.meta.url` inside vendored bundle files to `file:///...`,
  // which a Worker constructor can't load from an http(s) origin. Hosts
  // (the bench, wallet, etc.) can set `globalThis.__MIDEN_GPU_WORKER_URL`
  // to a usable URL before calling `TransactionProver.newGpuProver()`. The
  // override is also useful for production setups that serve the worker
  // from a CDN.
  if (
    typeof globalThis !== "undefined" &&
    typeof globalThis.__MIDEN_GPU_WORKER_URL === "string"
  ) {
    return globalThis.__MIDEN_GPU_WORKER_URL;
  }
  const path = `./${"workers"}/web-gpu-worker.module.js`;
  const candidate = new URL(path, import.meta.url);
  if (candidate.protocol === "file:") {
    throw new Error(
      "GPU worker URL resolved to file:// (Next.js + file: package quirk). " +
        "Set globalThis.__MIDEN_GPU_WORKER_URL to a fetchable URL before calling newGpuProver(). " +
        "Tip: in Next.js, copy `node_modules/@miden-sdk/miden-sdk/dist/gpu/workers/web-gpu-worker.module.js` " +
        "into `public/` and set `globalThis.__MIDEN_GPU_WORKER_URL = '/web-gpu-worker.module.js'`."
    );
  }
  return candidate;
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
