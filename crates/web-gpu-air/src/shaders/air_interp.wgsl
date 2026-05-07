// Tape-driven AIR constraint interpreter kernel.
//
// Each thread runs the entire tape against ONE row of input data and writes
// the alpha-folded accumulator (QuadFelt) into out[row].
//
// Bind group (8 storage buffers + 1 uniform — fits Chrome desktop's
// `maxStorageBuffersPerShaderStage = 8`):
//   @binding(0) tape:                array<vec4<u32>>   (each = Instruction)
//   @binding(1) inline_consts:       array<u32>
//   @binding(2) main_lde:            array<vec2<u32>>   (rows × main_width)
//   @binding(3) aux_lde:             array<vec4<u32>>   (rows × aux_width)
//   @binding(4) periodic_lde:        array<vec2<u32>>   (rows × num_periodic_columns)
//   @binding(5) ext_inputs:          array<vec4<u32>>   (concatenated:
//       randomness | permutation_values, offsets in `dims` uniform)
//   @binding(6) alpha_powers_global: array<vec4<u32>>
//   @binding(7) out:                 array<vec4<u32>>
//   @binding(8) dims (uniform):      KernelDims (widths + offsets)
//
// Public inputs and selectors are passed via the `dims` uniform (small,
// fixed-size). For the Phase 0b parity test we restrict to: no public
// values, no per-row selectors (selectors handled separately if needed).
//
// File expects goldilocks.wgsl + quadfelt.wgsl included before it.

// --- Opcodes (must match tape.rs constants) -------------------------------

const OP_LOAD_MAIN: u32              = 1u;
const OP_LOAD_AUX: u32               = 2u;
const OP_LOAD_PERIODIC: u32          = 3u;
const OP_LOAD_CONST_BASE: u32        = 4u;
const OP_LOAD_CONST_EXT: u32         = 5u;
const OP_LOAD_PUBLIC: u32            = 6u;
const OP_LOAD_RANDOMNESS: u32        = 7u;
const OP_LOAD_PERMUTATION_VALUE: u32 = 8u;
const OP_LOAD_IS_FIRST_ROW: u32      = 9u;
const OP_LOAD_IS_LAST_ROW: u32       = 10u;
const OP_LOAD_IS_TRANSITION: u32     = 11u;

const OP_ADD_BASE: u32 = 20u;
const OP_SUB_BASE: u32 = 21u;
const OP_MUL_BASE: u32 = 22u;
const OP_NEG_BASE: u32 = 23u;

const OP_ADD_EXT: u32      = 30u;
const OP_SUB_EXT: u32      = 31u;
const OP_MUL_EXT: u32      = 32u;
const OP_NEG_EXT: u32      = 33u;
const OP_MUL_BASE_EXT: u32 = 34u;
const OP_LIFT_BASE: u32    = 35u;

const OP_ASSERT_ZERO_BASE: u32 = 40u;
const OP_ASSERT_ZERO_EXT: u32  = 41u;

// --- Per-thread register file --------------------------------------------
//
// WGSL `var<private>` arrays must have compile-time-known size. We hardcode
// generous caps (well above the Phase 0a-measured 198 base / 123 ext for
// Miden's full AIR; 1024 of each = 8 KB + 16 KB = 24 KB — fits Apple
// Silicon's typical 32 KB private cap, exceeds the 16 KB spec minimum so
// the production kernel will need a smaller cap or a per-AIR generated
// const).

const MAX_BASE_REGS: u32 = 1024u;
const MAX_EXT_REGS: u32  = 1024u;

var<private> base_regs: array<vec2<u32>, 1024>;
var<private> ext_regs:  array<vec4<u32>, 1024>;

// --- Bind group ----------------------------------------------------------

@group(0) @binding(0) var<storage, read>       tape:                array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       inline_consts:       array<u32>;
@group(0) @binding(2) var<storage, read>       main_lde:            array<vec2<u32>>;
@group(0) @binding(3) var<storage, read>       aux_lde:             array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       periodic_lde:        array<vec2<u32>>;
@group(0) @binding(5) var<storage, read>       ext_inputs:          array<vec4<u32>>;
@group(0) @binding(6) var<storage, read>       alpha_powers_global: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> out:                 array<vec4<u32>>;

struct KernelDims {
    main_width: u32,
    aux_width: u32,
    num_periodic_columns: u32,
    rows: u32,
    randomness_offset: u32, // ext_inputs[ randomness_offset .. randomness_offset + num_randomness ]
    num_randomness: u32,
    permutation_values_offset: u32,
    num_permutation_values: u32,
};
@group(0) @binding(8) var<uniform> dims: KernelDims;

// --- Kernel main ---------------------------------------------------------

@compute @workgroup_size(64)
fn air_interp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) { return; }

    let main_pair_base = row * dims.main_width * 2u;     // current row + next row
    let aux_pair_base  = row * dims.aux_width * 2u;
    let periodic_base  = row * dims.num_periodic_columns;

    var acc: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u); // QuadFelt::ZERO

    let n_instr = arrayLength(&tape);
    for (var i: u32 = 0u; i < n_instr; i = i + 1u) {
        let instr = tape[i];
        let op = instr.x;
        let s1 = instr.y;
        let s2 = instr.z;
        let dst = instr.w;

        // Dispatch via if/else chain. WGSL switch is more restrictive
        // (case values must be const u32 literals; doesn't accept const-let
        // names), so the long if/else is simpler here.
        if (op == OP_LOAD_MAIN) {
            base_regs[dst] = main_lde[main_pair_base + s2 * dims.main_width + s1];
        } else if (op == OP_LOAD_AUX) {
            ext_regs[dst] = aux_lde[aux_pair_base + s2 * dims.aux_width + s1];
        } else if (op == OP_LOAD_PERIODIC) {
            base_regs[dst] = periodic_lde[periodic_base + s1];
        } else if (op == OP_LOAD_CONST_BASE) {
            base_regs[dst] = vec2<u32>(inline_consts[s1], inline_consts[s1 + 1u]);
        } else if (op == OP_LOAD_CONST_EXT) {
            ext_regs[dst] = vec4<u32>(
                inline_consts[s1],
                inline_consts[s1 + 1u],
                inline_consts[s1 + 2u],
                inline_consts[s1 + 3u],
            );
        } else if (op == OP_LOAD_PUBLIC) {
            // Public values not used by the parity test; tests skip this path.
            base_regs[dst] = vec2<u32>(0u, 0u);
        } else if (op == OP_LOAD_RANDOMNESS) {
            ext_regs[dst] = ext_inputs[dims.randomness_offset + s1];
        } else if (op == OP_LOAD_PERMUTATION_VALUE) {
            ext_regs[dst] = ext_inputs[dims.permutation_values_offset + s1];
        } else if (op == OP_LOAD_IS_FIRST_ROW) {
            base_regs[dst] = select(vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), row == 0u);
        } else if (op == OP_LOAD_IS_LAST_ROW) {
            base_regs[dst] = select(vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), row + 1u == dims.rows);
        } else if (op == OP_LOAD_IS_TRANSITION) {
            base_regs[dst] = select(vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), row + 1u != dims.rows);

        // ---- Base arithmetic ----
        } else if (op == OP_ADD_BASE) {
            base_regs[dst] = gl_add(base_regs[s1], base_regs[s2]);
        } else if (op == OP_SUB_BASE) {
            base_regs[dst] = gl_sub(base_regs[s1], base_regs[s2]);
        } else if (op == OP_MUL_BASE) {
            base_regs[dst] = gl_mul(base_regs[s1], base_regs[s2]);
        } else if (op == OP_NEG_BASE) {
            base_regs[dst] = gl_sub(vec2<u32>(0u, 0u), base_regs[s1]);

        // ---- Ext arithmetic ----
        } else if (op == OP_ADD_EXT) {
            ext_regs[dst] = qf_add(ext_regs[s1], ext_regs[s2]);
        } else if (op == OP_SUB_EXT) {
            ext_regs[dst] = qf_sub(ext_regs[s1], ext_regs[s2]);
        } else if (op == OP_MUL_EXT) {
            ext_regs[dst] = qf_mul(ext_regs[s1], ext_regs[s2]);
        } else if (op == OP_NEG_EXT) {
            ext_regs[dst] = qf_sub(vec4<u32>(0u, 0u, 0u, 0u), ext_regs[s1]);
        } else if (op == OP_MUL_BASE_EXT) {
            // dst (ext) = src1 (ext) * src2 (base)
            ext_regs[dst] = qf_mul_base(ext_regs[s1], base_regs[s2]);
        } else if (op == OP_LIFT_BASE) {
            ext_regs[dst] = qf_from_base(base_regs[s1]);

        // ---- Constraint emission ----
        } else if (op == OP_ASSERT_ZERO_BASE) {
            // acc += alpha_powers_global[s2] * lift(base_regs[s1])
            let v = qf_from_base(base_regs[s1]);
            acc = qf_add(acc, qf_mul(alpha_powers_global[s2], v));
        } else if (op == OP_ASSERT_ZERO_EXT) {
            acc = qf_add(acc, qf_mul(alpha_powers_global[s2], ext_regs[s1]));
        }
    }

    out[row] = acc;
}
