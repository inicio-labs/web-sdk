// Goldilocks (p = 2^64 - 2^32 + 1) arithmetic primitives for WGSL.
//
// All Goldilocks elements are stored as `vec2<u32>` with `(lo, hi)` word order:
//   x = x.y * 2^32 + x.x
//
// p_lo = 1, p_hi = 0xFFFFFFFF, EPSILON = 0xFFFFFFFF = 2^32 - 1
//
// The reduction follows Plonky3's `reduce128` pattern. WGSL has no widening
// 32×32→64 mul, so we synthesise it via 16-bit-half schoolbook (~10 inst /
// 32×32). A full 64×64→128 multiply is four 32×32 + carry plumbing
// (~50 inst), then `reduce128` knocks the result back to a canonical
// Goldilocks element.

const GL_EPSILON: u32 = 0xFFFFFFFFu;  // 2^32 - 1
const GL_P_HI: u32    = 0xFFFFFFFFu;  // p = (1, 0xFFFFFFFF)
const GL_P_LO: u32    = 0x00000001u;

// ---- 32×32 → 64-bit unsigned multiply ------------------------------------------------------

// Returns (lo, hi) such that lo + hi * 2^32 == a * b.
fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p_ll = a_lo * b_lo;  // bits 0..32  (max (2^16-1)^2 < 2^32)
    let p_lh = a_lo * b_hi;  // bits 16..48
    let p_hl = a_hi * b_lo;  // bits 16..48
    let p_hh = a_hi * b_hi;  // bits 32..64

    // mid = p_lh + p_hl  (may overflow u32 by 1 bit)
    let mid = p_lh + p_hl;
    let mid_carry = select(0u, 1u, mid < p_lh);

    // result_lo = p_ll + (mid << 16)
    let result_lo = p_ll + (mid << 16u);
    let lo_carry  = select(0u, 1u, result_lo < p_ll);

    // result_hi = p_hh + (mid >> 16) + (mid_carry << 16) + lo_carry
    let result_hi = p_hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;

    return vec2<u32>(result_lo, result_hi);
}

// ---- 64-bit primitives ---------------------------------------------------------------------

// Returns (a + b) mod 2^64 along with the 65-th bit carry.
// Output: (sum_lo, sum_hi, carry).
fn add_u64_carry(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo       = a.x + b.x;
    let lo_carry = select(0u, 1u, lo < a.x);
    let hi_part  = a.y + b.y;
    let hi_part_c = select(0u, 1u, hi_part < a.y);
    let hi       = hi_part + lo_carry;
    let hi_c2    = select(0u, 1u, hi < hi_part);
    return vec3<u32>(lo, hi, hi_part_c + hi_c2);
}

// Returns (a - b) mod 2^64 along with the borrow flag (1 if a < b).
fn sub_u64_borrow(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo        = a.x - b.x;
    let lo_borrow = select(0u, 1u, a.x < b.x);
    let hi_pre    = a.y - b.y;
    let hi_pre_b  = select(0u, 1u, a.y < b.y);
    let hi        = hi_pre - lo_borrow;
    let hi_b2     = select(0u, 1u, hi_pre < lo_borrow);
    return vec3<u32>(lo, hi, hi_pre_b + hi_b2);
}

// ---- 64×64 → 128-bit unsigned multiply -----------------------------------------------------

// Returns the four 32-bit limbs of the 128-bit product:
//   (limb0, limb1, limb2, limb3) such that
//   a * b == limb0 + limb1*2^32 + limb2*2^64 + limb3*2^96.
fn mul_u64_to_u128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p_ll = mul_u32(a.x, b.x);  // 64 bits at offset 0
    let p_lh = mul_u32(a.x, b.y);  // 64 bits at offset 32
    let p_hl = mul_u32(a.y, b.x);  // 64 bits at offset 32
    let p_hh = mul_u32(a.y, b.y);  // 64 bits at offset 64

    // limb0 = p_ll.x
    let limb0 = p_ll.x;

    // limb1 = p_ll.y + p_lh.x + p_hl.x   (may overflow into bit 32)
    let s0 = p_ll.y + p_lh.x;
    let s0_c = select(0u, 1u, s0 < p_ll.y);
    let limb1 = s0 + p_hl.x;
    let s1_c = select(0u, 1u, limb1 < s0);
    let carry_into_limb2 = s0_c + s1_c;

    // limb2 = p_lh.y + p_hl.y + p_hh.x + carry_into_limb2
    let s2 = p_lh.y + p_hl.y;
    let s2_c = select(0u, 1u, s2 < p_lh.y);
    let s3 = s2 + p_hh.x;
    let s3_c = select(0u, 1u, s3 < s2);
    let limb2 = s3 + carry_into_limb2;
    let s4_c = select(0u, 1u, limb2 < s3);
    let carry_into_limb3 = s2_c + s3_c + s4_c;

    // limb3 = p_hh.y + carry_into_limb3
    let limb3 = p_hh.y + carry_into_limb3;

    return vec4<u32>(limb0, limb1, limb2, limb3);
}

// ---- Goldilocks reduction ------------------------------------------------------------------

// Reduce a 128-bit value (limbs as packed in the vec4) to a canonical
// Goldilocks element in [0, p). Translation of Plonky3's reduce128.
fn gl_reduce_u128(x: vec4<u32>) -> vec2<u32> {
    let x_lo = vec2<u32>(x.x, x.y);  // bits 0..64
    // x_hi_hi = bits 96..128 = x.w
    // x_hi_lo = bits 64..96  = x.z
    let x_hi_hi: u32 = x.w;
    let x_hi_lo: u32 = x.z;

    // t0 = x_lo - x_hi_hi  (treat x_hi_hi as u64 = (x_hi_hi, 0))
    let sub0 = sub_u64_borrow(x_lo, vec2<u32>(x_hi_hi, 0u));
    var t0   = vec2<u32>(sub0.x, sub0.y);
    let borrow = sub0.z;

    // If borrow, t0 -= EPSILON  (subtract (0xFFFFFFFF, 0))
    if (borrow > 0u) {
        let sub1 = sub_u64_borrow(t0, vec2<u32>(GL_EPSILON, 0u));
        t0 = vec2<u32>(sub1.x, sub1.y);
        // Plonky3 ignores the second-order borrow — well-formed inputs (x < 2*p^2) won't underflow further.
    }

    // t1 = x_hi_lo * EPSILON  (u32 * u32 = u64)
    let t1 = mul_u32(x_hi_lo, GL_EPSILON);

    // res = t0 + t1  with carry
    let add0 = add_u64_carry(t0, t1);
    var final_lo = add0.x;
    var final_hi = add0.y;
    let carry = add0.z;

    // If carry, result += EPSILON
    if (carry > 0u) {
        let new_lo = final_lo + GL_EPSILON;
        let new_lo_c = select(0u, 1u, new_lo < final_lo);
        final_lo = new_lo;
        final_hi = final_hi + new_lo_c;
    }

    return gl_canonicalize(vec2<u32>(final_lo, final_hi));
}

// Final canonicalisation: ensure the result is in [0, p) by subtracting p
// if the value happens to be in [p, 2^64).
fn gl_canonicalize(x: vec2<u32>) -> vec2<u32> {
    // x >= p  iff  x.y > GL_P_HI  OR  (x.y == GL_P_HI AND x.x >= GL_P_LO)
    // Since GL_P_HI == 0xFFFFFFFF, x.y > GL_P_HI is impossible (x.y is u32),
    // so the first disjunct collapses to false. Just check the equal-and-ge case.
    let needs_sub = (x.y == GL_P_HI) && (x.x >= GL_P_LO);
    if (needs_sub) {
        // x - p = (x.x - 1, x.y - 0xFFFFFFFF) = (x.x - 1, 0)
        return vec2<u32>(x.x - GL_P_LO, x.y - GL_P_HI);
    }
    return x;
}

// ---- Public ops ----------------------------------------------------------------------------

fn gl_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s = add_u64_carry(a, b);
    var lo = s.x;
    var hi = s.y;
    let carry = s.z;
    if (carry > 0u) {
        // Overflow into bit 64: subtract 2^64 (free — already done) and add EPSILON.
        let new_lo = lo + GL_EPSILON;
        let new_c  = select(0u, 1u, new_lo < lo);
        lo = new_lo;
        hi = hi + new_c;
    }
    return gl_canonicalize(vec2<u32>(lo, hi));
}

fn gl_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let prod = mul_u64_to_u128(a, b);
    return gl_reduce_u128(prod);
}
