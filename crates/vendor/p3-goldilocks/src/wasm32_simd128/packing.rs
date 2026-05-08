//! WASM32 simd128 packed Goldilocks. WIDTH = 2 (one v128 register holds
//! two Goldilocks elements as i64x2 lanes).
//!
//! Phase P1.3: add, sub, neg primitives + standalone tests against scalar.
//! Phase P1.4 will add mul. Phase P1.5 will add the full PackedField trait
//! surface and wire `<Goldilocks as Field>::Packing` to point here.
//!
//! Translation table from `aarch64_neon` to wasm32 simd128:
//!   uint64x2_t                 → v128
//!   veorq_u64(a, b)            → v128_xor(a, b)
//!   vaddq_u64(a, b)            → i64x2_add(a, b)
//!   vsubq_u64(a, b)            → i64x2_sub(a, b)
//!   vcgtq_s64(a, b)            → i64x2_gt(a, b)
//!   vbicq_u64(a, b)            → v128_andnot(a, b)  (= a & !b)
//!   vshrq_n_u64::<32>(a)       → u64x2_shr(a, 32)
//!   vdupq_n_u64(x)             → u64x2_splat(x)
//!   vreinterpretq_s64_u64(x)   → identity (v128 is type-erased)

use alloc::vec::Vec;
use core::arch::wasm32::{
    i32x4_shuffle, i64x2_add, i64x2_extmul_low_u32x4, i64x2_gt, i64x2_shl, i64x2_shuffle,
    i64x2_sub, u64x2_shr, v128, v128_and, v128_andnot, v128_or, v128_xor,
};
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, PrimeField64,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::Goldilocks;

pub const WIDTH: usize = 2;

/// Equal to `2^32 - 1 = 2^64 mod P`.
const EPSILON: u64 = Goldilocks::ORDER_U64.wrapping_neg();

/// Vectorized wasm32-simd128 implementation of `Goldilocks` arithmetic.
///
/// `repr(transparent)` over `[Goldilocks; WIDTH]` so we can `transmute`
/// freely between `[Goldilocks; 2]`, `[u64; 2]`, and `v128`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksWasm32Simd128(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksWasm32Simd128 {
    #[inline]
    pub(crate) fn to_vector(self) -> v128 {
        unsafe { transmute(self) }
    }

    #[inline]
    pub(crate) fn from_vector(vector: v128) -> Self {
        unsafe { transmute(vector) }
    }

    #[inline]
    pub const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksWasm32Simd128 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

// ---- Constants used by the shifted-representation arithmetic ----

const SIGN_BIT: v128 =
    unsafe { transmute::<[u64; 2], v128>([0x8000_0000_0000_0000u64; WIDTH]) };
const SHIFTED_FIELD_ORDER: v128 = unsafe {
    transmute::<[u64; 2], v128>([Goldilocks::ORDER_U64 ^ 0x8000_0000_0000_0000u64; WIDTH])
};
const EPSILON_VEC: v128 = unsafe { transmute::<[u64; 2], v128>([EPSILON; WIDTH]) };

#[inline(always)]
fn shift(x: v128) -> v128 {
    v128_xor(x, SIGN_BIT)
}

/// If `x_s < SHIFTED_FIELD_ORDER` (signed comparison), add EPSILON to
/// canonicalize. The neon impl uses `vbicq_u64(EPSILON_VEC, mask)` =
/// `EPSILON_VEC & !mask`. wasm32's `v128_andnot(a, b) = a & !b` matches.
#[inline(always)]
fn canonicalize_s(x_s: v128) -> v128 {
    let mask = i64x2_gt(SHIFTED_FIELD_ORDER, x_s);
    let wrapback_amt = v128_andnot(EPSILON_VEC, mask);
    i64x2_add(x_s, wrapback_amt)
}

#[inline(always)]
fn add_no_double_overflow_64_64s_s(x: v128, y_s: v128) -> v128 {
    let res_wrapped_s = i64x2_add(x, y_s);
    // Overflow detected: y_s > res_wrapped_s (signed). On overflow, add EPSILON.
    let mask = i64x2_gt(y_s, res_wrapped_s);
    let wrapback_amt = u64x2_shr(mask, 32);
    i64x2_add(res_wrapped_s, wrapback_amt)
}

/// Goldilocks modular addition.
#[inline]
pub(crate) fn add(x: v128, y: v128) -> v128 {
    let y_s = shift(y);
    let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
    shift(res_s)
}

/// Goldilocks modular subtraction.
#[inline]
pub(crate) fn sub(x: v128, y: v128) -> v128 {
    let y_s = canonicalize_s(shift(y));
    let x_s = shift(x);
    let mask = i64x2_gt(y_s, x_s);
    let wrapback_amt = u64x2_shr(mask, 32);
    let res_wrapped = i64x2_sub(x_s, y_s);
    i64x2_sub(res_wrapped, wrapback_amt)
}

/// Goldilocks modular negation.
#[inline]
pub(crate) fn neg(y: v128) -> v128 {
    let y_s = shift(y);
    i64x2_sub(SHIFTED_FIELD_ORDER, canonicalize_s(y_s))
}

// ---- Multiplication: schoolbook 64×64 → 128 + Goldilocks reduction ----

/// Pack the low 32 bits of each u64 lane into u32 lanes 0 and 1.
/// Input  u32x4 view: [a0_lo, a0_hi, a1_lo, a1_hi]
/// Output u32x4 view: [a0_lo, a1_lo,    *,     *]
#[inline(always)]
fn lo32(a: v128) -> v128 {
    i32x4_shuffle::<0, 2, 0, 0>(a, a)
}

/// Pack the high 32 bits of each u64 lane into u32 lanes 0 and 1.
/// Input  u32x4 view: [a0_lo, a0_hi, a1_lo, a1_hi]
/// Output u32x4 view: [a0_hi, a1_hi,    *,     *]
#[inline(always)]
fn hi32(a: v128) -> v128 {
    i32x4_shuffle::<1, 3, 0, 0>(a, a)
}

/// 32×32 → 64-bit unsigned multiply, lane-aligned: returns
///   `[((a[0] as u32) * (b[0] as u32)) as u64, ((a[1] as u32) * (b[1] as u32)) as u64]`
/// where `a[i]`, `b[i]` are the low 32 bits of each u64 lane.
#[inline(always)]
fn mul_u32_lanes(a_packed: v128, b_packed: v128) -> v128 {
    // Both inputs have the desired u32 operands at u32x4 lanes 0 and 1.
    // i64x2_extmul_low_u32x4 reads exactly those lanes.
    i64x2_extmul_low_u32x4(a_packed, b_packed)
}

/// Full 64×64 → 128 multiply per lane. Returns `(hi, lo)` where the
/// 128-bit product per lane = `lo + hi * 2^64`. Translation of the AVX2
/// `mul64_64` (`x86_64_avx2/packing.rs:353-392`).
#[inline]
fn mul64_64(x: v128, y: v128) -> (v128, v128) {
    let x_lo = lo32(x);
    let x_hi = hi32(x);
    let y_lo = lo32(y);
    let y_hi = hi32(y);

    // Four pairwise 32×32 → 64 products.
    let ll = mul_u32_lanes(x_lo, y_lo); // x_lo * y_lo
    let lh = mul_u32_lanes(x_lo, y_hi); // x_lo * y_hi
    let hl = mul_u32_lanes(x_hi, y_lo);
    let hh = mul_u32_lanes(x_hi, y_hi);

    // Bignum addition (AVX2 algorithm verbatim):
    //   t0 = hl + (ll >> 32)              (cannot overflow: ≤ (2^32-1)^2 + (2^32-1) < 2^64)
    //   t1 = lh + (t0 & 0xFFFFFFFF)       (cannot overflow)
    //   t2 = hh + (t0 >> 32)              (cannot overflow)
    //   res_hi = t2 + (t1 >> 32)          (cannot overflow)
    //   res_lo = (ll & 0xFFFFFFFF) | ((t1 & 0xFFFFFFFF) << 32)
    let ll_hi = u64x2_shr(ll, 32);
    let t0 = i64x2_add(hl, ll_hi);
    let t0_lo = v128_and(t0, EPSILON_VEC);
    let t0_hi = u64x2_shr(t0, 32);
    let t1 = i64x2_add(lh, t0_lo);
    let t2 = i64x2_add(hh, t0_hi);
    let t1_hi = u64x2_shr(t1, 32);
    let res_hi = i64x2_add(t2, t1_hi);

    let ll_lo32 = v128_and(ll, EPSILON_VEC);
    let t1_lo32 = v128_and(t1, EPSILON_VEC);
    let t1_shifted = i64x2_shl(t1_lo32, 32);
    let res_lo = v128_or(ll_lo32, t1_shifted);

    (res_hi, res_lo)
}

/// Add a "small" pre-shifted Goldilocks lane: `x_s` shifted by 2^63,
/// `y ≤ 2^64 - 2^32 = 0xFFFFFFFF00000000`. Result is shifted by 2^63.
/// Translation of the AVX2 `add_small_64s_64_s`. Mask shift via 64-bit
/// rather than 32-bit comparison since wasm32 simd128's `i32x4_*` ops on
/// our 64-bit lanes would compare half-lanes — using `i64x2_gt` directly is
/// simpler and correct for all `y` magnitudes.
#[inline(always)]
fn add_small_64s_64_s(x_s: v128, y: v128) -> v128 {
    let res_wrapped_s = i64x2_add(x_s, y);
    let mask = i64x2_gt(x_s, res_wrapped_s); // signed compare, -1 if overflow
    let wrapback_amt = u64x2_shr(mask, 32); // 0xFFFFFFFF if overflow else 0
    i64x2_add(res_wrapped_s, wrapback_amt)
}

/// Subtract a "small" Goldilocks value. Mirror of `add_small_64s_64_s`.
#[inline(always)]
fn sub_small_64s_64_s(x_s: v128, y: v128) -> v128 {
    let res_wrapped_s = i64x2_sub(x_s, y);
    let mask = i64x2_gt(res_wrapped_s, x_s); // -1 if underflow
    let wrapback_amt = u64x2_shr(mask, 32);
    i64x2_sub(res_wrapped_s, wrapback_amt)
}

/// Reduce a 128-bit value `(hi, lo)` modulo Goldilocks order. Result fits
/// in 64 bits but may be ≥ FIELD_ORDER (subsequent canonicalize handles
/// that on demand). Translation of the AVX2 `reduce128`.
///
/// Uses `2^64 ≡ 2^32 - 1 (mod p)` and `2^96 ≡ -1 (mod p)`.
#[inline]
fn reduce128(hi: v128, lo: v128) -> v128 {
    let lo_s = shift(lo);
    // 2^96 ≡ -1, so the contribution of `hi_hi * 2^96` is `-hi_hi`.
    let hi_hi = u64x2_shr(hi, 32);
    let lo1_s = sub_small_64s_64_s(lo_s, hi_hi);

    // hi_lo32 * EPSILON  where EPSILON = 2^32 - 1.
    // AVX2 uses _mm256_mul_epu32(hi0, EPSILON) (low-32 multiply).
    // We compute it as (hi_lo32 << 32) - hi_lo32, avoiding a full multiply.
    // hi_lo32 ≤ 2^32 - 1, so (hi_lo32 << 32) ≤ 2^64 - 2^32, no overflow.
    let hi_lo32 = v128_and(hi, EPSILON_VEC);
    let hi_lo32_shifted = i64x2_shl(hi_lo32, 32);
    let t1 = i64x2_sub(hi_lo32_shifted, hi_lo32);

    // Result is at most (2^32 - 1)^2 < 2^64, so add_small_64s_64_s applies.
    let lo2_s = add_small_64s_64_s(lo1_s, t1);
    shift(lo2_s)
}

/// Goldilocks modular multiplication. Inputs may be ≥ FIELD_ORDER; output
/// is reduced into `[0, 2*FIELD_ORDER)` (canonicalize on demand).
#[inline]
pub(crate) fn mul(x: v128, y: v128) -> v128 {
    let (hi, lo) = mul64_64(x, y);
    reduce128(hi, lo)
}

// ---- Operator trait impls (delegate to the lane-level functions) --------

impl Add for PackedGoldilocksWasm32Simd128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(add(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGoldilocksWasm32Simd128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(sub(self.to_vector(), rhs.to_vector()))
    }
}

impl Neg for PackedGoldilocksWasm32Simd128 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_vector(neg(self.to_vector()))
    }
}

impl Mul for PackedGoldilocksWasm32Simd128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksWasm32Simd128);
impl_sub_assign!(PackedGoldilocksWasm32Simd128);
impl_mul_methods!(PackedGoldilocksWasm32Simd128);
ring_sum!(PackedGoldilocksWasm32Simd128);
impl_rng!(PackedGoldilocksWasm32Simd128);

// ---- PrimeCharacteristicRing + monomial impls --------------------------

/// Halve a vector of Goldilocks elements: `result = (input >> 1) + (P+1)/2 if input is odd`.
/// Mirrors the neon `halve` (line ~258 of aarch64_neon/packing.rs).
#[inline(always)]
fn halve(input: v128) -> v128 {
    use core::arch::wasm32::u64x2_splat;
    let one = u64x2_splat(1);
    let zero = u64x2_splat(0);
    let half_v = u64x2_splat(Goldilocks::ORDER_U64.div_ceil(2));
    let least_bit = v128_and(input, one);
    let t = u64x2_shr(input, 1);
    // neg_least_bit = 0 or -1 (broadcast least_bit to all bits within each lane).
    let neg_least_bit = i64x2_sub(zero, least_bit);
    let maybe_half = v128_and(half_v, neg_least_bit);
    i64x2_add(t, maybe_half)
}

#[inline(always)]
fn square(x: v128) -> v128 {
    // No specialized squaring path on simd128; fall through to mul.
    mul(x, x)
}

impl PrimeCharacteristicRing for PackedGoldilocksWasm32Simd128 {
    type PrimeSubfield = Goldilocks;

    const ZERO: Self = Self::broadcast(Goldilocks::ZERO);
    const ONE: Self = Self::broadcast(Goldilocks::ONE);
    const TWO: Self = Self::broadcast(Goldilocks::TWO);
    const NEG_ONE: Self = Self::broadcast(Goldilocks::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::from_vector(halve(self.to_vector()))
    }

    #[inline]
    fn square(&self) -> Self {
        Self::from_vector(square(self.to_vector()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }
}

impl InjectiveMonomial<7> for PackedGoldilocksWasm32Simd128 {}

impl PermutationMonomial<7> for PackedGoldilocksWasm32Simd128 {
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksWasm32Simd128, Goldilocks);
impl_sub_base_field!(PackedGoldilocksWasm32Simd128, Goldilocks);
impl_mul_base_field!(PackedGoldilocksWasm32Simd128, Goldilocks);
impl_div_methods!(PackedGoldilocksWasm32Simd128, Goldilocks);
impl_sum_prod_base_field!(PackedGoldilocksWasm32Simd128, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksWasm32Simd128 {
    // Match the neon BATCHED_LC_CHUNK; can be tuned empirically once the
    // wallet integration measures real prove time.
    const BATCHED_LC_CHUNK: usize = 2;
}

impl_packed_value!(PackedGoldilocksWasm32Simd128, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksWasm32Simd128 {
    type Scalar = Goldilocks;
}

/// Interleave two u64x2 vectors at the element level.
/// For block_len=1: [a0, a1] x [b0, b1] -> [a0, b0], [a1, b1]
#[inline]
pub fn interleave_u64(v0: v128, v1: v128) -> (v128, v128) {
    // i64x2_shuffle::<I0, I1>(a, b) selects lanes from concat(a; b), where
    // 0,1 are a's lanes and 2,3 are b's.
    let r0 = i64x2_shuffle::<0, 2>(v0, v1);
    let r1 = i64x2_shuffle::<1, 3>(v0, v1);
    (r0, r1)
}

unsafe impl PackedFieldPow2 for PackedGoldilocksWasm32Simd128 {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave_u64(v0, v1),
            2 => (v0, v1),
            _ => panic!("unsupported block length"),
        };
        (Self::from_vector(res0), Self::from_vector(res1))
    }
}

#[cfg(test)]
mod tests {
    //! Tests run on wasm32 only — that's the target where simd128 ops exist.
    //! Ran via `cargo test --target wasm32-unknown-unknown` with wasm-bindgen
    //! test runner. Native cargo test won't pick these up (the module is
    //! cfg-gated to wasm32).
    //!
    //! For Phase P1.3 we verify compile-on-target via
    //!   cargo build --target wasm32-unknown-unknown -p p3-goldilocks
    //! plus structural correctness via cargo check on host.

    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn pack(a: u64, b: u64) -> PackedGoldilocksWasm32Simd128 {
        PackedGoldilocksWasm32Simd128([Goldilocks::new(a), Goldilocks::new(b)])
    }

    #[test]
    fn add_zero_and_one() {
        // 0 + 0 = 0; 0 + 1 = 1; 1 + 1 = 2.
        let zeros = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ZERO);
        let ones = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ONE);
        let result = zeros.add(ones);
        assert_eq!(result.0, [Goldilocks::ONE, Goldilocks::ONE]);

        let twos = ones.add(ones);
        assert_eq!(twos.0, [Goldilocks::TWO, Goldilocks::TWO]);
    }

    #[test]
    fn sub_zero_and_one() {
        let ones = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ONE);
        let zeros = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ZERO);
        let result = ones.sub(ones);
        assert_eq!(result.0, [Goldilocks::ZERO, Goldilocks::ZERO]);

        // 0 - 1 wraps to P - 1.
        let result = zeros.sub(ones);
        assert_eq!(result.0, [Goldilocks::NEG_ONE, Goldilocks::NEG_ONE]);
    }

    #[test]
    fn neg_zero_and_one() {
        let zeros = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ZERO);
        assert_eq!(zeros.neg().0, [Goldilocks::ZERO, Goldilocks::ZERO]);

        let ones = PackedGoldilocksWasm32Simd128::broadcast(Goldilocks::ONE);
        assert_eq!(ones.neg().0, [Goldilocks::NEG_ONE, Goldilocks::NEG_ONE]);
    }

    /// Cross-check against scalar Goldilocks for several edge cases.
    #[test]
    fn add_edge_cases() {
        let cases: &[(u64, u64)] = &[
            (0, 0),
            (1, 1),
            (Goldilocks::ORDER_U64 - 1, 1), // wraps to 0
            (Goldilocks::ORDER_U64 - 1, Goldilocks::ORDER_U64 - 1),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0xFFFF_FFFF_FFFF_FFFF, 1), // > P input — both impls should reduce
        ];
        for &(a, b) in cases {
            let pkg = pack(a, b);
            let sum_pkg = pkg.add(pkg);
            let scalar_a = Goldilocks::new(a) + Goldilocks::new(a);
            let scalar_b = Goldilocks::new(b) + Goldilocks::new(b);
            assert_eq!(
                sum_pkg.0,
                [scalar_a, scalar_b],
                "add mismatch for ({a:#x}, {b:#x})"
            );
        }
    }
}
