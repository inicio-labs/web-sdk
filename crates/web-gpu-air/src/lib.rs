//! WebGPU-accelerated AIR constraint evaluator for the Miden web-sdk WASM prover.
//!
//! Phase 0a (current): measure-only modules:
//! - [`recorder`] — runs `<ProcessorAir as Air>::eval` against `SymbolicAirBuilder`
//!   to capture constraint expression DAGs.
//! - [`cse`] — common-subexpression elimination via structural hashing.
//! - [`alloc`] — register allocator (liveness-based).
//!
//! Phase 0b / Phase 2 will add `tape`, `cpu_interp`, and `gpu` modules under the
//! `real-gpu` feature.

pub mod alloc;
pub mod cpu_interp;
pub mod cse;
pub mod encode;
pub mod recorder;
pub mod tape;

#[cfg(feature = "real-gpu")]
pub mod gpu;
