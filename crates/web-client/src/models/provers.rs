use alloc::sync::Arc;
use core::time::Duration;

use miden_client::RemoteTransactionProver;
use miden_client::transaction::{
    LocalTransactionProver, ProvingOptions, TransactionProver as TransactionProverTrait,
};
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
use miden_protocol::transaction::{ProvenTransaction, TransactionInputs};
#[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
use miden_client::transaction::TransactionProverError;

/// Discriminator for which backing prover impl this `TransactionProver` wraps.
///
/// Stored as a struct field rather than recovered from the trait object — the underlying
/// `Arc<dyn TransactionProverTrait>` doesn't extend `std::any::Any`, so downcasting isn't
/// possible. The kind is set at construction time by each factory and read by `serialize()`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ProverKind {
    Local,
    Remote,
    #[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
    Gpu,
}

/// Wrapper over local or remote transaction proving backends.
#[wasm_bindgen]
#[derive(Clone)]
pub struct TransactionProver {
    prover: Arc<dyn TransactionProverTrait + Send + Sync>,
    // NOTE: `kind` MUST stay private (no `pub` modifier). wasm-bindgen would otherwise
    // attempt to expose it to JS and fail to compile because `ProverKind` isn't a
    // wasm-bindgen-compatible enum (it has cfg-gated variants).
    kind: ProverKind,
    endpoint: Option<String>,
    timeout: Option<Duration>,
}

#[wasm_bindgen]
impl TransactionProver {
    /// Creates a prover that uses the local proving backend.
    #[wasm_bindgen(js_name = "newLocalProver")]
    pub fn new_local_prover() -> TransactionProver {
        let local_prover = LocalTransactionProver::new(ProvingOptions::default());
        TransactionProver {
            prover: Arc::new(local_prover),
            kind: ProverKind::Local,
            endpoint: None,
            timeout: None,
        }
    }

    /// Creates a new remote transaction prover.
    ///
    /// Arguments:
    /// - `endpoint`: The URL of the remote prover.
    /// - `timeout_ms`: The timeout in milliseconds for the remote prover.
    #[wasm_bindgen(js_name = "newRemoteProver")]
    pub fn new_remote_prover(endpoint: &str, timeout_ms: Option<u64>) -> TransactionProver {
        let mut remote_prover = RemoteTransactionProver::new(endpoint);

        let timeout = if let Some(timeout) = timeout_ms {
            let timeout = Duration::from_millis(timeout);
            remote_prover = remote_prover.with_timeout(timeout);
            Some(timeout)
        } else {
            None
        };

        TransactionProver {
            prover: Arc::new(remote_prover),
            kind: ProverKind::Remote,
            endpoint: Some(endpoint.to_string()),
            timeout,
        }
    }

    /// Creates a prover backed by a WebGPU-accelerated DFT.
    ///
    /// Asynchronous because `WebGpuDft::new()` requests a `wgpu::Device` from the browser's
    /// `navigator.gpu` adapter. Returns an error on browsers without WebGPU support
    /// (Firefox stable, Safari < 18) — the bench's GPU tab surfaces this rather than
    /// falling back silently to CPU.
    #[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
    #[wasm_bindgen(js_name = "newGpuProver")]
    pub async fn new_gpu_prover() -> Result<TransactionProver, JsValue> {
        let dft = WebGpuTransactionProver::init_dft().await?;
        let inner = LocalTransactionProver::new(ProvingOptions::default());
        let prover = WebGpuTransactionProver { inner, dft };
        Ok(TransactionProver {
            prover: Arc::new(prover),
            kind: ProverKind::Gpu,
            endpoint: None,
            timeout: None,
        })
    }

    /// Serializes the prover configuration into a string descriptor.
    ///
    /// Format:
    /// - `"local"` for local prover
    /// - `"remote|{endpoint}"` for remote prover without timeout
    /// - `"remote|{endpoint}|{timeout_ms}"` for remote prover with timeout
    /// - `"gpu"` for the WebGPU prover (gpu-dft feature only)
    ///
    /// The GPU descriptor signals "re-init the device on load" — `wgpu::Device` handles
    /// aren't portable across serialize boundaries.
    pub fn serialize(&self) -> String {
        match self.kind {
            ProverKind::Local => "local".to_string(),
            ProverKind::Remote => match (&self.endpoint, &self.timeout) {
                (Some(ep), Some(timeout)) => {
                    let timeout_ms = u64::try_from(timeout.as_millis())
                        .expect("timeout was created from u64 milliseconds");
                    format!("remote|{ep}|{timeout_ms}")
                },
                (Some(ep), None) => format!("remote|{ep}"),
                (None, _) => {
                    // Defensive: a Remote-kind prover without an endpoint is malformed.
                    "local".to_string()
                },
            },
            #[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
            ProverKind::Gpu => "gpu".to_string(),
        }
    }

    /// Reconstructs a prover from its serialized descriptor.
    ///
    /// Async because re-initializing a GPU prover from `"gpu"` requires acquiring a fresh
    /// `wgpu::Device`. **Breaking change** from the previous sync signature; existing
    /// callers must add `.await`. Known caller migration: `miden-wallet/.../offscreen/main.ts:107`.
    ///
    /// Parses the format produced by `serialize()`:
    /// - `"local"` for local prover
    /// - `"remote|{endpoint}"` for remote prover without timeout
    /// - `"remote|{endpoint}|{timeout_ms}"` for remote prover with timeout
    /// - `"gpu"` for the WebGPU prover (returns an error if the build was made without
    ///   the `gpu-dft` feature, so a wallet rolling back from GPU build to ST/MT sees a
    ///   clear message rather than a generic "Invalid prover payload")
    pub async fn deserialize(payload: &str) -> Result<TransactionProver, JsValue> {
        if payload == "local" {
            return Ok(TransactionProver::new_local_prover());
        }

        #[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
        if payload == "gpu" {
            return TransactionProver::new_gpu_prover().await;
        }
        #[cfg(not(all(feature = "gpu-dft", target_arch = "wasm32")))]
        if payload == "gpu" {
            return Err(JsValue::from_str("GPU prover not available in this build"));
        }

        if let Some(rest) = payload.strip_prefix("remote|") {
            if rest.is_empty() {
                return Err(JsValue::from_str("Remote prover requires an endpoint"));
            }

            // Split on last `|` to extract optional timeout
            if let Some(last_pipe) = rest.rfind('|') {
                let endpoint = &rest[..last_pipe];
                let timeout_str = &rest[last_pipe + 1..];

                // Check if the suffix is a valid integer (timeout)
                if let Ok(timeout_ms) = timeout_str.parse::<u64>() {
                    return Ok(TransactionProver::new_remote_prover(
                        endpoint,
                        Some(timeout_ms),
                    ));
                }
            }

            // No valid timeout found, entire rest is the endpoint
            return Ok(TransactionProver::new_remote_prover(rest, None));
        }

        Err(JsValue::from_str(&format!(
            "Invalid prover payload: {payload}"
        )))
    }

    /// Returns the endpoint if this is a remote prover.
    pub fn endpoint(&self) -> Option<String> {
        self.endpoint.clone()
    }
}

impl TransactionProver {
    /// Returns the underlying proving trait object.
    pub fn get_prover(&self) -> Arc<dyn TransactionProverTrait + Send + Sync> {
        self.prover.clone()
    }
}

impl From<Arc<dyn TransactionProverTrait + Send + Sync>> for TransactionProver {
    fn from(prover: Arc<dyn TransactionProverTrait + Send + Sync>) -> Self {
        TransactionProver {
            prover,
            // We don't know the actual kind from a raw Arc — default to Local for serialize().
            // Callers that need an accurate descriptor should construct via the explicit
            // factory methods.
            kind: ProverKind::Local,
            endpoint: None,
            timeout: None,
        }
    }
}

// =============================================================================================
// WebGpuTransactionProver — wraps LocalTransactionProver + WebGpuDft, routes prove() through
// the wasm32-only `prove_with_dft` path so the inner STARK proof uses GPU NTTs.
// =============================================================================================

#[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
pub struct WebGpuTransactionProver {
    inner: LocalTransactionProver,
    dft: miden_web_gpu_dft::WebGpuDft,
}

#[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
impl WebGpuTransactionProver {
    /// Acquire the WebGPU device and install the dft handle into the thread-local global.
    ///
    /// In the CPU-stub build of `miden-web-gpu-dft` (default), this is sync + infallible.
    /// In the `real-gpu` build, this spawns a dedicated GPU worker, awaits its READY
    /// message, and constructs `WebGpuDft::new_with_sab(sab)` so the trait impls route
    /// through the SAB+Atomics protocol.
    async fn init_dft() -> Result<miden_web_gpu_dft::WebGpuDft, JsValue> {
        #[cfg(feature = "real-gpu")]
        let dft = {
            // SAB size from the GPU-DFT crate's protocol. The bootstrap allocates this,
            // posts it to the GPU worker, and resolves once the worker signals READY.
            let sab_js = bootstrap_gpu_worker(miden_web_gpu_dft::sab::SAB_SIZE as u32).await?;
            let sab: js_sys::SharedArrayBuffer = sab_js
                .dyn_into()
                .map_err(|_| JsValue::from_str("bootstrapGpuWorker did not return a SharedArrayBuffer"))?;
            miden_web_gpu_dft::WebGpuDft::new_with_sab(sab)
        };
        #[cfg(not(feature = "real-gpu"))]
        let dft = miden_web_gpu_dft::WebGpuDft::new();

        miden_web_gpu_dft::install_global(dft.clone());
        Ok(dft)
    }
}

#[cfg(all(feature = "gpu-dft", feature = "real-gpu", target_arch = "wasm32"))]
#[wasm_bindgen(module = "/js/gpu-bootstrap.js")]
extern "C" {
    /// Spawns the GPU worker, allocates a SharedArrayBuffer of size `sab_size`,
    /// posts the SAB to the worker, awaits its READY response, and returns the
    /// SAB. See `crates/web-client/js/gpu-bootstrap.js`.
    #[wasm_bindgen(js_name = bootstrapGpuWorker, catch)]
    async fn bootstrap_gpu_worker(sab_size: u32) -> Result<JsValue, JsValue>;
}

#[cfg(all(feature = "gpu-dft", target_arch = "wasm32"))]
#[async_trait::async_trait(?Send)]
impl miden_client::transaction::TransactionProver for WebGpuTransactionProver {
    async fn prove(
        &self,
        witness: TransactionInputs,
    ) -> Result<ProvenTransaction, TransactionProverError> {
        self.inner.prove_with_dft(witness, self.dft.clone()).await
    }
}
