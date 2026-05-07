//! Phase P1.4: validate the algorithm used by the wasm32 simd128 packed
//! Goldilocks `mul`. The `scalar_twin` module in p3-goldilocks implements
//! the same algorithm using u64 ops (the simd128 version is a direct
//! op-for-op translation), so testing the u64 twin against canonical
//! scalar Goldilocks proves the algorithm is correct on all targets.
//!
//! The hardware codegen of the simd128 version is independently validated
//! by `wasm-bindgen-test` (set up in Phase P1.5).

use miden_crypto::{Felt, field::PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_goldilocks::wasm32_simd128::scalar_twin::mul_scalar_twin;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn rand_in_range<R: Rng>(rng: &mut R) -> u64 {
    loop {
        let v: u64 = rng.random();
        if v < Goldilocks::ORDER_U64 {
            return v;
        }
    }
}

fn canonical_mul(a: u64, b: u64) -> u64 {
    let result = Felt::new(a) * Felt::new(b);
    result.as_canonical_u64()
}

fn canonicalize(v: u64) -> u64 {
    if v >= Goldilocks::ORDER_U64 {
        v - Goldilocks::ORDER_U64
    } else {
        v
    }
}

#[test]
fn mul_scalar_twin_zero() {
    for v in [0u64, 1, 12345, Goldilocks::ORDER_U64 - 1] {
        assert_eq!(canonicalize(mul_scalar_twin(0, v)), 0);
        assert_eq!(canonicalize(mul_scalar_twin(v, 0)), 0);
    }
}

#[test]
fn mul_scalar_twin_one() {
    for v in [0u64, 1, 12345, Goldilocks::ORDER_U64 - 1] {
        assert_eq!(canonicalize(mul_scalar_twin(1, v)), v);
        assert_eq!(canonicalize(mul_scalar_twin(v, 1)), v);
    }
}

#[test]
fn mul_scalar_twin_edge_cases() {
    let cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 1),
        (Goldilocks::ORDER_U64 - 1, Goldilocks::ORDER_U64 - 1),
        (0xFFFF_FFFF, 0xFFFF_FFFF),
        (0xFFFF_FFFF_FFFF_FFFF, 1),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        (Goldilocks::ORDER_U64, Goldilocks::ORDER_U64),
        (0x12345678_9ABCDEF0, 0xFEDCBA98_76543210),
    ];
    for &(a, b) in cases {
        let twin = canonicalize(mul_scalar_twin(a, b));
        let canonical = canonical_mul(a, b);
        assert_eq!(
            twin, canonical,
            "mismatch on ({a:#x}, {b:#x}): twin={twin:#x} canonical={canonical:#x}"
        );
    }
}

#[test]
fn mul_scalar_twin_random_10000() {
    let mut rng = StdRng::seed_from_u64(0xBEEF_F00D);
    let mut mismatches = Vec::new();
    for _ in 0..10_000 {
        let a = rand_in_range(&mut rng);
        let b = rand_in_range(&mut rng);
        let twin = canonicalize(mul_scalar_twin(a, b));
        let canonical = canonical_mul(a, b);
        if twin != canonical {
            mismatches.push((a, b, twin, canonical));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches in 10k random pairs (sample: {:?})",
        mismatches.len(),
        mismatches.iter().take(5).collect::<Vec<_>>()
    );
}
