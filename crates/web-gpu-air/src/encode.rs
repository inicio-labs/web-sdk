//! Byte encoding helpers for shipping `Felt` / `QuadFelt` values to the
//! WGSL kernels and back. The kernel encodes:
//! - `Felt` as `vec2<u32>` little-endian limbs `(lo, hi)` such that
//!   `value == lo + hi * 2^32`.
//! - `QuadFelt = (a0, a1)` as `vec4<u32>` where `(.xy)` is `a0` and `(.zw)`
//!   is `a1`.
//!
//! On the Rust side both representations are 8-byte and 16-byte aligned-u32
//! arrays. We use `[u32; 2]` and `[u32; 4]` so they can be packed straight
//! into a `bytemuck::Pod`-friendly `wgpu::Buffer`.

use miden_crypto::{Felt, field::BasedVectorSpace};

use crate::recorder::QuadFelt;

/// Encode a Felt as the WGSL `vec2<u32>` representation.
#[inline]
pub fn felt_to_limbs(f: Felt) -> [u32; 2] {
    let v = f.as_canonical_u64();
    [v as u32, (v >> 32) as u32]
}

/// Decode a Felt from its WGSL `vec2<u32>` representation.
#[inline]
pub fn limbs_to_felt(limbs: [u32; 2]) -> Felt {
    let v = (limbs[0] as u64) | ((limbs[1] as u64) << 32);
    Felt::new(v)
}

/// Encode a QuadFelt as the WGSL `vec4<u32>` representation: `(.xy = a0, .zw = a1)`.
#[inline]
pub fn quadfelt_to_limbs(q: QuadFelt) -> [u32; 4] {
    let coeffs: &[Felt] = q.as_basis_coefficients_slice();
    debug_assert_eq!(coeffs.len(), 2);
    let a0 = felt_to_limbs(coeffs[0]);
    let a1 = felt_to_limbs(coeffs[1]);
    [a0[0], a0[1], a1[0], a1[1]]
}

/// Decode a QuadFelt from its WGSL `vec4<u32>` representation.
#[inline]
pub fn limbs_to_quadfelt(limbs: [u32; 4]) -> QuadFelt {
    let a0 = limbs_to_felt([limbs[0], limbs[1]]);
    let a1 = limbs_to_felt([limbs[2], limbs[3]]);
    // BinomialExtensionField: a0 + a1·X
    QuadFelt::new([a0, a1])
}

/// Random sampling helpers (visible to all tests in the crate). Kept here
/// rather than in a `tests` submodule so cross-module tests (e.g. WGSL parity
/// tests in another module) can reuse them.
#[cfg(test)]
pub(crate) mod testing {
    use miden_crypto::Felt;
    use rand::Rng;

    use super::QuadFelt;

    pub fn random_felt<R: Rng>(rng: &mut R) -> Felt {
        // Sample uniformly from [0, p). p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001.
        loop {
            let v: u64 = rng.random();
            if v < 0xFFFFFFFF_00000001u64 {
                return Felt::new(v);
            }
        }
    }

    pub fn random_quadfelt<R: Rng>(rng: &mut R) -> QuadFelt {
        QuadFelt::new([random_felt(rng), random_felt(rng)])
    }
}

#[cfg(test)]
mod tests {
    use miden_crypto::field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::testing::{random_felt, random_quadfelt};
    use super::*;

    #[test]
    fn felt_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        for _ in 0..1000 {
            let f = random_felt(&mut rng);
            assert_eq!(limbs_to_felt(felt_to_limbs(f)), f);
        }
    }

    #[test]
    fn quadfelt_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xBABE);
        for _ in 0..1000 {
            let q = random_quadfelt(&mut rng);
            assert_eq!(limbs_to_quadfelt(quadfelt_to_limbs(q)), q);
        }
    }

    #[test]
    fn felt_zero_one() {
        let zero = Felt::ZERO;
        let one = Felt::ONE;
        assert_eq!(felt_to_limbs(zero), [0, 0]);
        assert_eq!(felt_to_limbs(one), [1, 0]);
        assert_eq!(limbs_to_felt([0, 0]), zero);
        assert_eq!(limbs_to_felt([1, 0]), one);
    }

    #[test]
    fn quadfelt_zero_one() {
        let zero = QuadFelt::ZERO;
        let one = QuadFelt::ONE;
        assert_eq!(quadfelt_to_limbs(zero), [0, 0, 0, 0]);
        // ONE = 1 + 0·X
        assert_eq!(quadfelt_to_limbs(one), [1, 0, 0, 0]);
    }
}
