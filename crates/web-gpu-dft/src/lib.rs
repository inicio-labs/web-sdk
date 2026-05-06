//! WebGPU-accelerated DFT/NTT backend for the Miden WASM prover.
//!
//! This crate implements `p3_dft::TwoAdicSubgroupDft<Felt>` over WebGPU compute, so the
//! browser-side STARK prover can route LDE / FRI NTTs through the GPU instead of the
//! single-threaded WASM CPU path. The integration point is Phase 2's
//! `miden_prover::prove_with_dft` and the wallet-side `LocalTransactionProver::prove_with_dft`
//! (also wasm32-only).
//!
//! ## Status
//!
//! The crate currently ships a **CPU-delegating stub** under the default feature set: it
//! implements the trait by forwarding to `Radix2DitParallel<Felt>`. This validates the
//! end-to-end wiring through Phase 4 (the bench's GPU tab produces a verifying proof) without
//! committing to the full WGSL shader development. The real GPU kernels land behind the
//! `real-gpu` feature flag in a follow-up.
//!
//! ## Default-bound workaround
//!
//! `TwoAdicSubgroupDft` requires `Clone + Default`. We can't construct a real `WebGpuDft` from
//! a sync `Default::default()` because device init is async. The workaround is a thread-local
//! `OnceCell<WebGpuDft>` populated by [`install_global`] from the async init path
//! (`TransactionProver::newGpuProver` in Phase 4); `Default::default()` clones from there.
//! On the explicit dispatch path (caller passes a `WebGpuDft` instance into `prove_with_dft`),
//! `Default::default()` is never invoked at runtime — the thread-local exists only to satisfy
//! the trait bound.

#![cfg_attr(not(test), warn(missing_docs))]

extern crate alloc;

mod dft;

pub use dft::WebGpuDft;

use miden_crypto::Felt;
use miden_crypto::stark::dft::Radix2DitParallel;

/// Inner state of a `WebGpuDft` handle.
///
/// In the CPU-delegating stub, this just holds a `Radix2DitParallel<Felt>`. In the real-GPU
/// build, this will own the `wgpu::Device`, `wgpu::Queue`, compiled `ComputePipeline`s, the
/// twiddle-factor cache, and the scratch buffer pool.
pub(crate) struct WebGpuDftInner {
    pub(crate) cpu_fallback: Radix2DitParallel<Felt>,
}

impl Default for WebGpuDftInner {
    fn default() -> Self {
        Self {
            cpu_fallback: Radix2DitParallel::default(),
        }
    }
}

/// Errors that can occur during `WebGpuDft::new()` async device acquisition.
#[derive(Debug, thiserror::Error)]
pub enum GpuInitError {
    /// The browser does not expose `navigator.gpu` or `requestAdapter()` returned null.
    #[error("WebGPU adapter unavailable")]
    AdapterUnavailable,
    /// `requestDevice()` failed.
    #[error("WebGPU device init failed: {0}")]
    DeviceInit(alloc::string::String),
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static GLOBAL_GPU: core::cell::OnceCell<WebGpuDft> = const { core::cell::OnceCell::new() };
}

/// Install a `WebGpuDft` as the thread-local global so `Default::default()` can clone it.
///
/// Call this once from `TransactionProver::newGpuProver` after `WebGpuDft::new().await`
/// succeeds. It's a no-op on second call (OnceCell semantics).
#[cfg(target_arch = "wasm32")]
pub fn install_global(dft: WebGpuDft) {
    GLOBAL_GPU.with(|c| {
        let _ = c.set(dft);
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn global() -> Option<WebGpuDft> {
    GLOBAL_GPU.with(|c| c.get().cloned())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn global() -> Option<WebGpuDft> {
    None
}

/// Stub for the no-op install path on non-wasm targets so callers don't need to gate.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_global(_dft: WebGpuDft) {}
