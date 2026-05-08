//! The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]

extern crate alloc;

mod extension;
mod goldilocks;
mod mds;
mod poseidon2;

pub use goldilocks::*;
pub use mds::*;
pub use poseidon2::*;

pub mod poseidon1;

#[cfg(target_arch = "aarch64")]
mod aarch64_neon;

#[cfg(target_arch = "aarch64")]
pub use aarch64_neon::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod x86_64_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use x86_64_avx512::*;

// wasm32_simd128 backend: the `packing` submodule (the actual SIMD impl)
// is gated on `target_arch = "wasm32"` + `target_feature = "simd128"`, but
// the `scalar_twin` submodule is host-runnable so we can exercise the
// algorithm under regular `cargo test` without a wasm32 runtime. The
// outer module is therefore always available and gates the inner pieces
// itself.
pub mod wasm32_simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32_simd128::*;
