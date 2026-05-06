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
// Gate the wgpu integration on `real-gpu`. Compiles on both native and
// wasm32 — the kernel methods have async cores that work on either target,
// plus native-only sync wrappers (gated inside gpu.rs) that pollster the
// async cores. Wasm32 callers must drive the async methods inside an async
// context (e.g. inside the GPU worker's spawn_local-driven command loop).
#[cfg(feature = "real-gpu")]
mod gpu;
// SharedArrayBuffer protocol between the prover thread and the GPU worker.
// Same layout used by both sides; gated on real-gpu only because the SAB
// dance only matters when GPU is active.
#[cfg(feature = "real-gpu")]
pub mod sab;
// Synchronous GPU client (wasm32 only) — what `WebGpuDft`'s sync trait
// methods route through to dispatch GPU work to the worker via SAB+Atomics.
#[cfg(all(feature = "real-gpu", target_arch = "wasm32"))]
pub mod wasm_client;
// Async GPU worker entry (wasm32 only) — runs inside the dedicated GPU
// Web Worker, processes commands from the SAB, dispatches async wgpu work.
#[cfg(all(feature = "real-gpu", target_arch = "wasm32"))]
pub mod wasm_worker;

pub use dft::WebGpuDft;
#[cfg(feature = "real-gpu")]
pub use gpu::WgpuContext;

use miden_crypto::Felt;
use miden_crypto::stark::dft::Radix2DitParallel;

/// Inner state of a `WebGpuDft` handle.
///
/// Always carries a CPU fallback (`Radix2DitParallel<Felt>`) for the methods we don't yet
/// have a GPU kernel for, plus — when the `real-gpu` feature is on AND we're on a target
/// where the sync trait impl can drive GPU work to completion (currently native only via
/// `wgpu::Device::poll(Wait)`) — a `WgpuContext` for the GPU dispatches.
///
/// On wasm32, GPU readback is fundamentally async (`mapAsync` returns a Promise driven by
/// the JS event loop), so the sync trait method can't directly call into the GPU without
/// a SharedArrayBuffer + Atomics.wait bridge to a dedicated GPU worker. That bridge lives
/// behind an additional `real-gpu-wasm` feature (separate substep). Until that's wired,
/// the wasm32 build with `real-gpu` falls through to the CPU fallback.
pub(crate) struct WebGpuDftInner {
    pub(crate) cpu_fallback: Radix2DitParallel<Felt>,
    /// Native-only: a real wgpu device + queue, used directly by the trait
    /// impls via pollster::block_on in the sync method body.
    #[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
    pub(crate) wgpu_ctx: Option<gpu::WgpuContext>,
    /// Wasm32-only: SAB-based sync client to a dedicated GPU worker. The
    /// trait impls block on Atomics::wait while the worker runs the async
    /// wgpu operations on its own thread.
    #[cfg(all(feature = "real-gpu", target_arch = "wasm32"))]
    pub(crate) gpu_client: Option<wasm_client::GpuClient>,
}

impl Default for WebGpuDftInner {
    fn default() -> Self {
        Self {
            cpu_fallback: Radix2DitParallel::default(),
            #[cfg(all(feature = "real-gpu", not(target_arch = "wasm32")))]
            wgpu_ctx: None,
            #[cfg(all(feature = "real-gpu", target_arch = "wasm32"))]
            gpu_client: None,
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
