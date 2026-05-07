//! Flat instruction-tape encoding for the AIR constraint interpreter.
//!
//! Each instruction is 4 u32s `(op, src1, src2, dst)`. Operand interpretation
//! depends on the opcode (see the `OP_*` constants below). The tape is
//! uploaded to the WGSL kernel as a `vec<u32>` storage buffer; the
//! interpreter walks it linearly per row.
//!
//! Constants (Felt or QuadFelt) referenced by `OP_LOAD_CONST_*` live in a
//! separate `inline_consts` `vec<u32>` buffer. Each Felt occupies 2 u32s,
//! each QuadFelt 4 u32s. The `src1` field of a LoadConst instruction is the
//! u32 offset into this buffer.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use miden_crypto::Felt;

use crate::alloc::{AllocResult, NodeKind};
use crate::cse::{BaseLeafKey, CseGraph, CseNode, ExtLeafKey, NodeId};
use crate::encode::{felt_to_limbs, quadfelt_to_limbs};
use crate::recorder::QuadFelt;

// --- Opcode constants ----------------------------------------------------------

// Loads (produce a value into dst). For loads, `src1` and `src2` carry
// opcode-specific operand info as documented at each opcode.

/// `dst = main_lde[col=src1, row_off=src2]` (src2 ∈ {0, 1}).
pub const OP_LOAD_MAIN: u32 = 1;
/// `dst = aux_lde[col=src1, row_off=src2]` (src2 ∈ {0, 1}); dst is ext.
pub const OP_LOAD_AUX: u32 = 2;
/// `dst = periodic_lde[col=src1]` for the current row.
pub const OP_LOAD_PERIODIC: u32 = 3;
/// `dst = inline_consts[src1 .. src1+2]` decoded as a Felt (vec2<u32>).
pub const OP_LOAD_CONST_BASE: u32 = 4;
/// `dst = inline_consts[src1 .. src1+4]` decoded as a QuadFelt (vec4<u32>).
pub const OP_LOAD_CONST_EXT: u32 = 5;
/// `dst = public_values[src1]` (base).
pub const OP_LOAD_PUBLIC: u32 = 6;
/// `dst = randomness[src1]` (ext).
pub const OP_LOAD_RANDOMNESS: u32 = 7;
/// `dst = permutation_values[src1]` (ext).
pub const OP_LOAD_PERMUTATION_VALUE: u32 = 8;
/// `dst = is_first_row` selector (base, 0 or 1).
pub const OP_LOAD_IS_FIRST_ROW: u32 = 9;
/// `dst = is_last_row` selector (base, 0 or 1).
pub const OP_LOAD_IS_LAST_ROW: u32 = 10;
/// `dst = is_transition` selector (base, 0 or 1).
pub const OP_LOAD_IS_TRANSITION: u32 = 11;

// Base-field arithmetic.
pub const OP_ADD_BASE: u32 = 20;
pub const OP_SUB_BASE: u32 = 21;
pub const OP_MUL_BASE: u32 = 22;
pub const OP_NEG_BASE: u32 = 23;

// Extension-field arithmetic.
pub const OP_ADD_EXT: u32 = 30;
pub const OP_SUB_EXT: u32 = 31;
pub const OP_MUL_EXT: u32 = 32;
pub const OP_NEG_EXT: u32 = 33;
/// `dst (ext) = a (ext) * b (base)`. Base × ext → ext.
pub const OP_MUL_BASE_EXT: u32 = 34;
/// `dst (ext) = lift(src1 (base) → ext)`. Just sets ext.0 = base, ext.1 = 0.
pub const OP_LIFT_BASE: u32 = 35;

// Constraint emission. `src1` = register holding the value, `src2` = the
// global constraint index k. The kernel folds via
// `acc += alpha_powers_global[k] * value_at(src1)`.
pub const OP_ASSERT_ZERO_BASE: u32 = 40;
pub const OP_ASSERT_ZERO_EXT: u32 = 41;

/// One tape instruction. Layout matches `vec4<u32>` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct Instruction {
    pub op: u32,
    pub src1: u32,
    pub src2: u32,
    pub dst: u32,
}

impl Instruction {
    pub fn new(op: u32, src1: u32, src2: u32, dst: u32) -> Self {
        Self {
            op,
            src1,
            src2,
            dst,
        }
    }
}

/// Lowered AIR constraint program. Self-describing: contains the
/// instruction tape, the constants pool, the register file dimensions, and
/// the constraint count for `alpha_powers_global` sizing.
#[derive(Clone, Debug)]
pub struct AirTape {
    pub instructions: Vec<Instruction>,
    /// Packed u32s: each Felt occupies 2 entries (lo, hi); each QuadFelt
    /// occupies 4 entries (a0_lo, a0_hi, a1_lo, a1_hi). The `src1` field of
    /// a LoadConst instruction is the u32 offset (multiple of 2 for base,
    /// multiple of 4 for ext).
    pub inline_consts: Vec<u32>,
    /// Number of distinct base registers the kernel needs (sized for the
    /// per-thread `var<private>` array).
    pub base_reg_count: u32,
    /// Number of distinct ext registers.
    pub ext_reg_count: u32,
    /// Total constraint count = `alpha_powers_global` length.
    pub constraint_count: u32,
}

/// Lower a CSE'd DAG + register allocation into a flat instruction tape.
///
/// The recorded `RecordedAir` is needed to map root nodes back to their
/// global constraint index (the `k` that drives the alpha-fold). Root order
/// is determined by `RecordedAir::constraint_layout` so the tape's
/// `AssertZero` k-tags match the CPU folder's expected alpha-power ordering.
pub fn lower(
    graph: &CseGraph,
    alloc: &AllocResult,
    constraint_count: u32,
) -> AirTape {
    // Map: NodeId of a root → its global k (constraint index). Built from
    // `roots_base` and `roots_ext` in graph order; the global index of
    // graph.roots_base[i] is the same `k` used by the CPU folder for
    // `constraints_base[i]`.
    let mut root_global_k: HashMap<NodeId, u32> = HashMap::new();
    // Roots_base + roots_ext together cover all 0..constraint_count.
    // The convention: global index k corresponds to the order in which
    // the recorder emitted the constraint via assert_zero / assert_zero_ext.
    // We don't have that order here directly — we get base_indices and
    // ext_indices from ConstraintLayout, but for the tape we use the order
    // in which the recorder produced base_constraints[i] / ext_constraints[i].
    //
    // For Phase 0b parity: the simplest convention is "global k = i where
    // graph.roots_base[i] is the i-th base root, and ext roots come after".
    // The CPU oracle in cpu_interp.rs uses the same convention. Phase 1's
    // equivalence test against the CPU folder checks this matches.
    for (i, &root) in graph.roots_base.iter().enumerate() {
        root_global_k.insert(root, i as u32);
    }
    let base_count = graph.roots_base.len() as u32;
    for (i, &root) in graph.roots_ext.iter().enumerate() {
        root_global_k.insert(root, base_count + i as u32);
    }

    let mut instrs: Vec<Instruction> = Vec::with_capacity(graph.nodes.len());
    let mut consts: Vec<u32> = Vec::new();

    // Cache: leaf-key → const offset (so identical constants share a slot).
    let mut const_cache_base: HashMap<u64, u32> = HashMap::new();
    let mut const_cache_ext: HashMap<[u64; 2], u32> = HashMap::new();

    let intern_const_base = |consts: &mut Vec<u32>,
                             cache: &mut HashMap<u64, u32>,
                             v: u64|
     -> u32 {
        if let Some(&off) = cache.get(&v) {
            return off;
        }
        let off = consts.len() as u32;
        let limbs = felt_to_limbs(Felt::new(v));
        consts.push(limbs[0]);
        consts.push(limbs[1]);
        cache.insert(v, off);
        off
    };

    let intern_const_ext = |consts: &mut Vec<u32>,
                            cache: &mut HashMap<[u64; 2], u32>,
                            v: [u64; 2]|
     -> u32 {
        if let Some(&off) = cache.get(&v) {
            return off;
        }
        let off = consts.len() as u32;
        consts.push(v[0] as u32);
        consts.push((v[0] >> 32) as u32);
        consts.push(v[1] as u32);
        consts.push((v[1] >> 32) as u32);
        cache.insert(v, off);
        off
    };

    for (node_id, node) in graph.nodes.iter().enumerate() {
        let dst = alloc.reg_for_node[node_id];
        let kind = alloc.kind_for_node[node_id];
        match node {
            // ---- Base leaves ----
            CseNode::LeafBase(BaseLeafKey::Variable {
                entry_kind,
                entry_offset,
                index,
            }) => {
                match entry_kind {
                    // Preprocessed (kind 0) — Miden has none, but handle anyway.
                    // We treat it as a load from a notional preprocessed buffer; for now,
                    // unsupported (no Miden constraint should reference it).
                    0 => panic!(
                        "preprocessed columns not supported yet: idx={index} offset={entry_offset}"
                    ),
                    // Main (kind 1)
                    1 => instrs.push(Instruction::new(
                        OP_LOAD_MAIN,
                        *index as u32,
                        *entry_offset as u32,
                        dst,
                    )),
                    // Periodic (kind 2)
                    2 => instrs.push(Instruction::new(
                        OP_LOAD_PERIODIC,
                        *index as u32,
                        0,
                        dst,
                    )),
                    // Public (kind 3)
                    3 => instrs.push(Instruction::new(
                        OP_LOAD_PUBLIC,
                        *index as u32,
                        0,
                        dst,
                    )),
                    other => panic!("unknown BaseEntry kind {other}"),
                }
            }
            CseNode::LeafBase(BaseLeafKey::IsFirstRow) => {
                instrs.push(Instruction::new(OP_LOAD_IS_FIRST_ROW, 0, 0, dst));
            }
            CseNode::LeafBase(BaseLeafKey::IsLastRow) => {
                instrs.push(Instruction::new(OP_LOAD_IS_LAST_ROW, 0, 0, dst));
            }
            CseNode::LeafBase(BaseLeafKey::IsTransition) => {
                instrs.push(Instruction::new(OP_LOAD_IS_TRANSITION, 0, 0, dst));
            }
            CseNode::LeafBase(BaseLeafKey::Constant(c)) => {
                let off = intern_const_base(&mut consts, &mut const_cache_base, *c);
                instrs.push(Instruction::new(OP_LOAD_CONST_BASE, off, 0, dst));
            }

            // ---- Ext leaves ----
            CseNode::LeafExt(ExtLeafKey::Base(base_node)) => {
                let src_base_reg = alloc.reg_for_node[*base_node as usize];
                instrs.push(Instruction::new(OP_LIFT_BASE, src_base_reg, 0, dst));
            }
            CseNode::LeafExt(ExtLeafKey::ExtVariable {
                entry_kind,
                entry_offset,
                index,
            }) => match entry_kind {
                // Permutation (kind 0)
                0 => instrs.push(Instruction::new(
                    OP_LOAD_AUX,
                    *index as u32,
                    *entry_offset as u32,
                    dst,
                )),
                // Challenge (kind 1)
                1 => instrs.push(Instruction::new(
                    OP_LOAD_RANDOMNESS,
                    *index as u32,
                    0,
                    dst,
                )),
                // PermutationValue (kind 2)
                2 => instrs.push(Instruction::new(
                    OP_LOAD_PERMUTATION_VALUE,
                    *index as u32,
                    0,
                    dst,
                )),
                other => panic!("unknown ExtEntry kind {other}"),
            },
            CseNode::LeafExt(ExtLeafKey::ExtConstant(c)) => {
                let off = intern_const_ext(&mut consts, &mut const_cache_ext, *c);
                instrs.push(Instruction::new(OP_LOAD_CONST_EXT, off, 0, dst));
            }

            // ---- Base arithmetic ----
            CseNode::AddBase(x, y) => instrs.push(arith_node(
                OP_ADD_BASE, *x, *y, dst, alloc,
            )),
            CseNode::SubBase(x, y) => instrs.push(arith_node(
                OP_SUB_BASE, *x, *y, dst, alloc,
            )),
            CseNode::MulBase(x, y) => instrs.push(arith_node(
                OP_MUL_BASE, *x, *y, dst, alloc,
            )),
            CseNode::NegBase(x) => {
                let xr = alloc.reg_for_node[*x as usize];
                instrs.push(Instruction::new(OP_NEG_BASE, xr, 0, dst));
            }

            // ---- Ext arithmetic ----
            CseNode::AddExt(x, y) => instrs.push(arith_node(
                OP_ADD_EXT, *x, *y, dst, alloc,
            )),
            CseNode::SubExt(x, y) => instrs.push(arith_node(
                OP_SUB_EXT, *x, *y, dst, alloc,
            )),
            CseNode::MulExt(x, y) => instrs.push(arith_node(
                OP_MUL_EXT, *x, *y, dst, alloc,
            )),
            CseNode::NegExt(x) => {
                let xr = alloc.reg_for_node[*x as usize];
                instrs.push(Instruction::new(OP_NEG_EXT, xr, 0, dst));
            }
        }

        // Emit AssertZero* if this node is a root.
        if let Some(&k) = root_global_k.get(&(node_id as NodeId)) {
            let op = match kind {
                NodeKind::Base => OP_ASSERT_ZERO_BASE,
                NodeKind::Ext => OP_ASSERT_ZERO_EXT,
            };
            instrs.push(Instruction::new(op, dst, k, 0));
        }
    }

    AirTape {
        instructions: instrs,
        inline_consts: consts,
        base_reg_count: alloc.base_reg_count,
        ext_reg_count: alloc.ext_reg_count,
        constraint_count,
    }
}

#[inline]
fn arith_node(op: u32, x: NodeId, y: NodeId, dst: u32, alloc: &AllocResult) -> Instruction {
    let xr = alloc.reg_for_node[x as usize];
    let yr = alloc.reg_for_node[y as usize];
    Instruction::new(op, xr, yr, dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_layout_is_4_u32s() {
        assert_eq!(core::mem::size_of::<Instruction>(), 16);
        assert_eq!(core::mem::align_of::<Instruction>(), 4);
    }

    #[test]
    fn lower_processor_air_smoke() {
        // End-to-end: record → CSE → alloc → lower. Just confirm the tape
        // emits, has reasonable sizes, and has at least one AssertZero.
        let recorded = crate::recorder::record_processor_air();
        let graph = CseGraph::from_recorded(&recorded);
        let alloc = crate::alloc::allocate(&graph);
        let total_constraints = (recorded.constraints_base.len()
            + recorded.constraints_ext.len()) as u32;
        let tape = lower(&graph, &alloc, total_constraints);

        let n_assert_zero_base = tape
            .instructions
            .iter()
            .filter(|i| i.op == OP_ASSERT_ZERO_BASE)
            .count();
        let n_assert_zero_ext = tape
            .instructions
            .iter()
            .filter(|i| i.op == OP_ASSERT_ZERO_EXT)
            .count();

        eprintln!(
            "ProcessorAir tape\n\
             ────────────────────────────────────────\n\
             Instructions:        {}\n\
             Inline consts (u32): {} (~{} Felt-equivalent)\n\
             Base regs:           {}\n\
             Ext regs:            {}\n\
             Constraint count:    {}\n\
             AssertZeroBase:      {} (matches roots_base = {})\n\
             AssertZeroExt:       {} (matches roots_ext  = {})\n",
            tape.instructions.len(),
            tape.inline_consts.len(),
            tape.inline_consts.len() / 2,
            tape.base_reg_count,
            tape.ext_reg_count,
            tape.constraint_count,
            n_assert_zero_base,
            recorded.constraints_base.len(),
            n_assert_zero_ext,
            recorded.constraints_ext.len(),
        );

        assert_eq!(n_assert_zero_base, recorded.constraints_base.len());
        assert_eq!(n_assert_zero_ext, recorded.constraints_ext.len());
        assert_eq!(
            tape.constraint_count,
            (recorded.constraints_base.len() + recorded.constraints_ext.len()) as u32,
        );
    }
}
