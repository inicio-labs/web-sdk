//! Host-side u64 implementation of the *same* algorithm used by the
//! wasm32 simd128 packing module (`mul64_64` + `reduce128`). This module
//! is host-runnable (no `cfg(target_arch = "wasm32")` gate) so we can
//! exercise the algorithm under regular `cargo test` without a wasm32
//! runtime.
//!
//! The simd128 module is a direct op-for-op translation of this u64 code:
//! each line over `v128` (e.g. `i64x2_add(a, b)`) corresponds to a `u64`
//! op (e.g. `a.wrapping_add(b)`). If this u64 version produces the same
//! field element as canonical scalar Goldilocks for randomized inputs,
//! the simd128 version does too (modulo the WGSL→hardware step, which is
//! validated by `wasm-bindgen-test` once we set that up).

const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

#[inline]
fn shift(x: u64) -> u64 {
    x ^ SIGN_BIT
}

#[inline]
fn add_small_64s_64_s(x_s: u64, y: u64) -> u64 {
    let res_wrapped_s = x_s.wrapping_add(y);
    let mask: u64 = if (x_s as i64) > (res_wrapped_s as i64) {
        u64::MAX
    } else {
        0
    };
    let wrapback_amt = mask >> 32;
    res_wrapped_s.wrapping_add(wrapback_amt)
}

#[inline]
fn sub_small_64s_64_s(x_s: u64, y: u64) -> u64 {
    let res_wrapped_s = x_s.wrapping_sub(y);
    let mask: u64 = if (res_wrapped_s as i64) > (x_s as i64) {
        u64::MAX
    } else {
        0
    };
    let wrapback_amt = mask >> 32;
    res_wrapped_s.wrapping_sub(wrapback_amt)
}

#[inline]
fn mul64_64(x: u64, y: u64) -> (u64, u64) {
    let x_lo = x & 0xFFFF_FFFF;
    let x_hi = x >> 32;
    let y_lo = y & 0xFFFF_FFFF;
    let y_hi = y >> 32;

    let ll = x_lo * y_lo;
    let lh = x_lo * y_hi;
    let hl = x_hi * y_lo;
    let hh = x_hi * y_hi;

    let ll_hi = ll >> 32;
    let t0 = hl.wrapping_add(ll_hi);
    let t0_lo = t0 & 0xFFFF_FFFF;
    let t0_hi = t0 >> 32;
    let t1 = lh.wrapping_add(t0_lo);
    let t2 = hh.wrapping_add(t0_hi);
    let t1_hi = t1 >> 32;
    let res_hi = t2.wrapping_add(t1_hi);
    let res_lo = (ll & 0xFFFF_FFFF) | (t1 << 32);
    (res_hi, res_lo)
}

#[inline]
fn reduce128(hi: u64, lo: u64) -> u64 {
    let lo_s = shift(lo);
    let hi_hi = hi >> 32;
    let lo1_s = sub_small_64s_64_s(lo_s, hi_hi);

    let hi_lo32 = hi & 0xFFFF_FFFF;
    let t1 = (hi_lo32 << 32).wrapping_sub(hi_lo32);

    let lo2_s = add_small_64s_64_s(lo1_s, t1);
    shift(lo2_s)
}

/// `x * y mod p` via the same algorithm used by the simd128 packing
/// module. Output may be ≥ FIELD_ORDER (canonicalize on demand).
pub fn mul_scalar_twin(x: u64, y: u64) -> u64 {
    let (hi, lo) = mul64_64(x, y);
    reduce128(hi, lo)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    fn rand_in_range<R: Rng>(rng: &mut R) -> u64 {
        // Sample uniformly from [0, p).
        loop {
            let v: u64 = rng.random();
            if v < Goldilocks::ORDER_U64 {
                return v;
            }
        }
    }

    /// Canonical scalar Goldilocks mul for comparison.
    fn canonical_mul(a: u64, b: u64) -> u64 {
        let result = Goldilocks::new(a) * Goldilocks::new(b);
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
    fn mul_zero() {
        for v in [0u64, 1, 12345, Goldilocks::ORDER_U64 - 1] {
            assert_eq!(canonicalize(mul_scalar_twin(0, v)), 0);
            assert_eq!(canonicalize(mul_scalar_twin(v, 0)), 0);
        }
    }

    #[test]
    fn mul_one() {
        for v in [0u64, 1, 12345, Goldilocks::ORDER_U64 - 1] {
            assert_eq!(canonicalize(mul_scalar_twin(1, v)), v);
            assert_eq!(canonicalize(mul_scalar_twin(v, 1)), v);
        }
    }

    #[test]
    fn mul_edge_cases() {
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
    fn mul_random_parity_10000() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_F00D);
        for _ in 0..10_000 {
            let a = rand_in_range(&mut rng);
            let b = rand_in_range(&mut rng);
            let twin = canonicalize(mul_scalar_twin(a, b));
            let canonical = canonical_mul(a, b);
            if twin != canonical {
                panic!(
                    "mismatch: a={a:#x} b={b:#x} twin={twin:#x} canonical={canonical:#x}"
                );
            }
        }
    }
}
