//! WASM32-simd128 packed Goldilocks backend.
//!
//! Mirrors the structure of `aarch64_neon` (WIDTH = 2, repr(transparent)
//! over `[Goldilocks; 2]`). Packed ops use `core::arch::wasm32::*`
//! intrinsics; reduction follows the same shifted-representation pattern
//! the neon impl uses, since wasm32 simd128 has the matching i64x2
//! comparison + shift ops.

// Host-runnable u64 implementation of the same algorithm — exists on all
// targets so `cargo test` (native, in any consumer crate) exercises the
// core math correctness. Public so tests in other crates can verify it.
pub mod scalar_twin;

// The actual wasm32 simd128 implementation. Gated on the right target_arch
// + target_feature so it's only compiled where the v128 ops exist.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod packing;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use packing::*;
