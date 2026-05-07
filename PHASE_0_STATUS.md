# GPU AIR Phase 0 — Final Status

**Verdict: HARD FAIL on perf gate. Pivot to wasm-simd128 + GPU Merkle.**

## Phase 0a — Structural feasibility (GREEN)

| Risk | Threshold | Measured | Status |
|------|-----------|----------|--------|
| (b) `<ProcessorAir>::eval` against `SymbolicAirBuilder` | non-empty | 441 base + 24 ext = 465 constraints | ✅ |
| (c) `MainTraceRow::align_to` with `SymbolicVariable<Felt>` | no panic | size 1704 B, align 8 B | ✅ |
| (e1) post-CSE node count | <5k comfortable, <10k hard | **3788** | ✅ comfortable |
| (e2) peak private memory | <8 KB comfortable, <16 KB hard | **2.6 KB** (198 base + 123 ext regs) | ✅ comfortable |

## Phase 0b — Correctness (GREEN)

- `qf_mul` WGSL byte-for-byte matches canonical `BinomialExtensionField` (1024 random pairs).
- Toy AIR end-to-end: WGSL tape interpreter byte-for-byte matches CPU oracle on 64 rows of random main + aux data with both base and ext constraints.

## Phase 0b — Perf (HARD FAIL)

Synthetic 5000-instruction tape × 1M rows on Apple Silicon (Metal):

```
MT CPU:   1220 ms (rayon, 10 threads, release)  →  4.30 GOPS aggregate
GPU warm:  765 ms (incl. dispatch + readback)   →  6.85 GOPS
Speedup:   1.60×  (target ≥ 4×, hard fail < 2×)
```

Both CPU and GPU at compute ceiling; GPU's parallelism advantage is offset by the
~50-instruction-per-Goldilocks-mul WGSL emulation cost. This matches the plan's
Risk #7 honest perf envelope ("4× ceiling, more likely 8-11s middle bucket"); we
landed below even the middle bucket.

Per the Phase 0 decision gate, this triggers `< 2× over MT CPU → abandon GPU AIR.
Pivot to wasm-simd128 + GPU Merkle.`

## What was built (all committed)

- `crates/web-gpu-air/` — full Phase 0 prototype, tape interpreter, CSE,
  register allocator, WGSL kernel, parity tests. **22 tests pass under
  `--features real-gpu`**, two `#[ignore]` perf benches available via
  `cargo test ... -- --ignored --nocapture`.
- All Phase 0a measurements documented in commit messages; perf numbers in
  Unit 8 commit.

## Pivot scaffolded

- `/Users/celrisen/miden/p3-goldilocks/` — local fork of p3-goldilocks v0.5.2
  patched into the workspace.
- `wasm32_simd128/{mod,packing}.rs` — backend skeleton with add, sub, neg
  primitives (P1.3). Compiles clean on `wasm32-unknown-unknown` with
  `-Ctarget-feature=+simd128`.
- `Cargo.toml` patch entry pointing the workspace at the fork.

## Pivot progress — wasm-simd128 packed Goldilocks

**P1.1 prep** ✅ Workspace patched to local fork at `/Users/celrisen/miden/p3-goldilocks/`.

**P1.2 study** ✅ Translation table from aarch64_neon documented in commits.

**P1.3 add/sub/neg** ✅ Ported from neon's shifted-representation. Compiles on `wasm32+simd128`.

**P1.4 mul + reduce128** ✅ Schoolbook 64×64→128 via `i64x2_extmul_low_u32x4` × 4 partial products + Goldilocks reduction (verbatim port of AVX2 `mul64_64` + `reduce128`). Algorithmic correctness verified by 10,000-pair host-side u64 twin test against canonical scalar Goldilocks.

**P1.5 trait surface + `Field::Packing` wiring** ✅
- Full `PrimeCharacteristicRing`, `Algebra<Goldilocks>`, `PackedField`,
  `PackedFieldPow2`, `InjectiveMonomial<7>`, `PermutationMonomial<7>` ported.
- `<Goldilocks as Field>::Packing` wired to `PackedGoldilocksWasm32Simd128`
  under `cfg(all(target_arch = "wasm32", target_feature = "simd128"))`.
- Subsequent Plonky3 packed-eval calls now use width=2 instead of width=1.
- Compiles clean on `wasm32-unknown-unknown` with `+simd128`.

**P1.6 — Measure** (in progress at end of session):
- Build the wallet's `gpu-wasm` variant. The `par_loop_eval` span metadata
  should now show `width=2` (vs current `width=1`).
- Bench full prove duration vs ~13s baseline. Plan estimate: ~3s saved.

## GPU Merkle (still to plan)

Independent ~1-2 week project. ~2-2.5s saved. Plan recommendation: do
wasm-simd128 first (smaller effort, faster ROI), then GPU Merkle after.
Combined target: ~5s shaved, prove from ~13s to ~8s.
