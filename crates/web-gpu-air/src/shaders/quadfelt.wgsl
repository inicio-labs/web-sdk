// QuadFelt arithmetic for WGSL. QuadFelt = Goldilocks degree-2 binomial
// extension with W = 7 (irreducible factor x^2 - 7 over Goldilocks; verified
// in p3-goldilocks-0.5.2/src/extension.rs:13).
//
// A QuadFelt (a0, a1) is encoded as `vec4<u32>`:
//   .xy = a0 (Felt as vec2<u32> lo/hi)
//   .zw = a1
//
// This file expects `goldilocks.wgsl` to be included before it (provides
// gl_add, gl_sub, gl_mul). The kernel that uses both shaders is responsible
// for concatenating the two source files in the right order.

const QF_W_LO: u32 = 7u;   // W = 7 in low limb
const QF_W_HI: u32 = 0u;   // High limb of W is zero (W < 2^32)

// Multiplication: (a0 + a1·X)·(b0 + b1·X) where X² = W = 7.
// Result = (a0·b0 + W·a1·b1, a0·b1 + a1·b0).
//
// Cost: 4 gl_mul + 1 gl_mul-by-constant-W + 2 gl_add ≈ ~5 base muls + 2 base
// adds. Per-row, this dominates each ext-field constraint evaluation.
fn qf_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let a0 = a.xy;
    let a1 = a.zw;
    let b0 = b.xy;
    let b1 = b.zw;

    // Real part: a0·b0 + W·a1·b1
    let a0b0 = gl_mul(a0, b0);
    let a1b1 = gl_mul(a1, b1);
    let w    = vec2<u32>(QF_W_LO, QF_W_HI);
    let wa1b1 = gl_mul(w, a1b1);
    let r0 = gl_add(a0b0, wa1b1);

    // Imag part: a0·b1 + a1·b0
    let a0b1 = gl_mul(a0, b1);
    let a1b0 = gl_mul(a1, b0);
    let r1 = gl_add(a0b1, a1b0);

    return vec4<u32>(r0.x, r0.y, r1.x, r1.y);
}

// Addition: component-wise Goldilocks add.
fn qf_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let r0 = gl_add(a.xy, b.xy);
    let r1 = gl_add(a.zw, b.zw);
    return vec4<u32>(r0.x, r0.y, r1.x, r1.y);
}

// Subtraction: component-wise Goldilocks sub.
fn qf_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let r0 = gl_sub(a.xy, b.xy);
    let r1 = gl_sub(a.zw, b.zw);
    return vec4<u32>(r0.x, r0.y, r1.x, r1.y);
}

// Multiply ext × base: a is QuadFelt, b is Felt. Result = (a0·b, a1·b).
fn qf_mul_base(a: vec4<u32>, b: vec2<u32>) -> vec4<u32> {
    let r0 = gl_mul(a.xy, b);
    let r1 = gl_mul(a.zw, b);
    return vec4<u32>(r0.x, r0.y, r1.x, r1.y);
}

// Helper: lift a Felt to QuadFelt (a0 = b, a1 = 0).
fn qf_from_base(b: vec2<u32>) -> vec4<u32> {
    return vec4<u32>(b.x, b.y, 0u, 0u);
}
