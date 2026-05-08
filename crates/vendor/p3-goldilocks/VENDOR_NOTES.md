# p3-goldilocks (vendored fork)

This is a Miden-local fork of `p3-goldilocks` (Plonky3 Goldilocks crate),
originally imported from crates.io v0.5.2.

## Why vendored

We added `wasm32_simd128/` — a 2-lane SIMD-packed Goldilocks implementation
using hand-written `core::arch::wasm32::*` intrinsics. It wires up to Plonky3's
`PackedField` trait surface, so on `wasm32+simd128` builds `Goldilocks::Packing
= PackedGoldilocksWasm32Simd128` (width=2 instead of width=1).

Without this, browser-side wasm proves run scalar Goldilocks even when
`+simd128` is enabled at the rustflags level — Plonky3 upstream has no
wasm32 SIMD module.

## Upstream status

Not yet upstreamed to `Plonky3/Plonky3`. Once they accept a PR adding the
wasm32 module, this vendored copy can be replaced with the released version
and the `[patch.crates-io]` entry in the workspace `Cargo.toml` dropped.

## Updating

To pull a new upstream Plonky3 release into this fork:
1. Diff our `wasm32_simd128/` module against the new upstream `goldilocks`.
2. Re-import the upstream files (everything except `wasm32_simd128/`).
3. Re-apply our `Goldilocks::Packing` cfg gate to `goldilocks.rs`.
4. Run `cargo test --target wasm32-unknown-unknown` to verify.
