//! CPU tape interpreter — the correctness oracle for the GPU kernel.
//!
//! Given an `AirTape` + per-row data, produces a single QuadFelt
//! accumulator by walking the instructions and folding each AssertZero
//! into `acc += alpha_powers_global[k] * value_at(reg)`.
//!
//! Phase 0b Unit 7b: this interpreter is the byte-for-byte oracle the
//! WGSL kernel is verified against. Phase 1 Unit 7 will additionally
//! verify it against `ProverConstraintFolder::finalize_constraints` on the
//! real Miden AIR.

use miden_crypto::{
    Felt,
    field::{ExtensionField, PrimeCharacteristicRing},
};

use crate::encode::{limbs_to_felt, limbs_to_quadfelt};
use crate::recorder::QuadFelt;
use crate::tape::{
    AirTape, OP_ADD_BASE, OP_ADD_EXT, OP_ASSERT_ZERO_BASE, OP_ASSERT_ZERO_EXT, OP_LIFT_BASE,
    OP_LOAD_AUX, OP_LOAD_CONST_BASE, OP_LOAD_CONST_EXT, OP_LOAD_IS_FIRST_ROW,
    OP_LOAD_IS_LAST_ROW, OP_LOAD_IS_TRANSITION, OP_LOAD_MAIN, OP_LOAD_PERIODIC,
    OP_LOAD_PERMUTATION_VALUE, OP_LOAD_PUBLIC, OP_LOAD_RANDOMNESS, OP_MUL_BASE, OP_MUL_BASE_EXT,
    OP_MUL_EXT, OP_NEG_BASE, OP_NEG_EXT, OP_SUB_BASE, OP_SUB_EXT,
};

/// Per-row inputs the interpreter reads from. Slices borrowed from upper
/// caller (so we can run the same interpreter over each row of an LDE
/// without copying).
pub struct RowInputs<'a> {
    /// Length: 2 × main_width (current row, then next row).
    pub main_pair: &'a [Felt],
    /// Length: 2 × aux_width (current row, then next row).
    pub aux_pair: &'a [QuadFelt],
    /// Length: num_periodic_columns. Values for the current row.
    pub periodic: &'a [Felt],
    pub public_values: &'a [Felt],
    /// Permutation challenges (length: num_permutation_challenges).
    pub randomness: &'a [QuadFelt],
    /// Permutation values (length: num_permutation_values).
    pub permutation_values: &'a [QuadFelt],
    /// Selectors (each 0 or 1).
    pub is_first_row: Felt,
    pub is_last_row: Felt,
    pub is_transition: Felt,
    /// Length: tape.constraint_count. alpha_powers_global[k] is the
    /// alpha-power weight for constraint k. The CPU folder uses a
    /// per-bucket layout; the GPU evaluator (and this oracle) uses a
    /// single global vector indexed by k.
    pub alpha_powers_global: &'a [QuadFelt],
}

impl<'a> RowInputs<'a> {
    fn main_width(&self) -> usize {
        self.main_pair.len() / 2
    }
    fn aux_width(&self) -> usize {
        self.aux_pair.len() / 2
    }
}

/// Run the tape against `inputs` and return the alpha-folded accumulator.
pub fn run_tape(tape: &AirTape, inputs: &RowInputs<'_>) -> QuadFelt {
    let mut base_regs: Vec<Felt> = vec![Felt::ZERO; tape.base_reg_count as usize];
    let mut ext_regs: Vec<QuadFelt> = vec![QuadFelt::ZERO; tape.ext_reg_count as usize];
    let mut acc = QuadFelt::ZERO;

    let main_width = inputs.main_width();
    let aux_width = inputs.aux_width();

    for instr in &tape.instructions {
        let op = instr.op;
        let src1 = instr.src1 as usize;
        let src2 = instr.src2 as usize;
        let dst = instr.dst as usize;

        match op {
            // ---- Base loads ----
            OP_LOAD_MAIN => {
                // src1 = column, src2 = row offset (0 = current, 1 = next)
                let row_base = src2 * main_width;
                base_regs[dst] = inputs.main_pair[row_base + src1];
            }
            OP_LOAD_AUX => {
                let row_base = src2 * aux_width;
                ext_regs[dst] = inputs.aux_pair[row_base + src1];
            }
            OP_LOAD_PERIODIC => {
                base_regs[dst] = inputs.periodic[src1];
            }
            OP_LOAD_CONST_BASE => {
                let lo = tape.inline_consts[src1];
                let hi = tape.inline_consts[src1 + 1];
                base_regs[dst] = limbs_to_felt([lo, hi]);
            }
            OP_LOAD_CONST_EXT => {
                let limbs = [
                    tape.inline_consts[src1],
                    tape.inline_consts[src1 + 1],
                    tape.inline_consts[src1 + 2],
                    tape.inline_consts[src1 + 3],
                ];
                ext_regs[dst] = limbs_to_quadfelt(limbs);
            }
            OP_LOAD_PUBLIC => {
                base_regs[dst] = inputs.public_values[src1];
            }
            OP_LOAD_RANDOMNESS => {
                ext_regs[dst] = inputs.randomness[src1];
            }
            OP_LOAD_PERMUTATION_VALUE => {
                ext_regs[dst] = inputs.permutation_values[src1];
            }
            OP_LOAD_IS_FIRST_ROW => {
                base_regs[dst] = inputs.is_first_row;
            }
            OP_LOAD_IS_LAST_ROW => {
                base_regs[dst] = inputs.is_last_row;
            }
            OP_LOAD_IS_TRANSITION => {
                base_regs[dst] = inputs.is_transition;
            }

            // ---- Base arithmetic ----
            OP_ADD_BASE => base_regs[dst] = base_regs[src1] + base_regs[src2],
            OP_SUB_BASE => base_regs[dst] = base_regs[src1] - base_regs[src2],
            OP_MUL_BASE => base_regs[dst] = base_regs[src1] * base_regs[src2],
            OP_NEG_BASE => base_regs[dst] = -base_regs[src1],

            // ---- Ext arithmetic ----
            OP_ADD_EXT => ext_regs[dst] = ext_regs[src1] + ext_regs[src2],
            OP_SUB_EXT => ext_regs[dst] = ext_regs[src1] - ext_regs[src2],
            OP_MUL_EXT => ext_regs[dst] = ext_regs[src1] * ext_regs[src2],
            OP_NEG_EXT => ext_regs[dst] = -ext_regs[src1],
            OP_MUL_BASE_EXT => {
                // dst (ext) = src1 (ext) * src2 (base)
                ext_regs[dst] = ext_regs[src1] * base_regs[src2];
            }
            OP_LIFT_BASE => {
                // dst (ext) = lift(src1 (base)) — base value as ext.
                ext_regs[dst] = QuadFelt::from(base_regs[src1]);
            }

            // ---- Constraint emission ----
            OP_ASSERT_ZERO_BASE => {
                // src1 = base reg, src2 = global constraint index k
                let v = base_regs[src1];
                let k = src2;
                acc += inputs.alpha_powers_global[k] * v;
            }
            OP_ASSERT_ZERO_EXT => {
                let v = ext_regs[src1];
                let k = src2;
                acc += inputs.alpha_powers_global[k] * v;
            }

            other => panic!("cpu_interp: unknown opcode {other}"),
        }
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{AirTape, Instruction};

    fn empty_tape() -> AirTape {
        AirTape {
            instructions: Vec::new(),
            inline_consts: Vec::new(),
            base_reg_count: 0,
            ext_reg_count: 0,
            constraint_count: 0,
        }
    }

    fn empty_inputs<'a>() -> RowInputs<'a> {
        RowInputs {
            main_pair: &[],
            aux_pair: &[],
            periodic: &[],
            public_values: &[],
            randomness: &[],
            permutation_values: &[],
            is_first_row: Felt::ZERO,
            is_last_row: Felt::ZERO,
            is_transition: Felt::ZERO,
            alpha_powers_global: &[],
        }
    }

    #[test]
    fn empty_tape_returns_zero() {
        let tape = empty_tape();
        let inputs = empty_inputs();
        assert_eq!(run_tape(&tape, &inputs), QuadFelt::ZERO);
    }

    /// Tape: load main column 0, assert_zero. Constraint 0 has alpha-power 1.
    /// Result: alpha_global[0] * main[0].
    #[test]
    fn one_base_constraint() {
        let tape = AirTape {
            instructions: vec![
                Instruction::new(OP_LOAD_MAIN, 0, 0, 0), // base_reg[0] = main[0]
                Instruction::new(OP_ASSERT_ZERO_BASE, 0, 0, 0), // acc += alpha[0] * base_reg[0]
            ],
            inline_consts: Vec::new(),
            base_reg_count: 1,
            ext_reg_count: 0,
            constraint_count: 1,
        };
        let main_pair = vec![Felt::from_u32(7), Felt::ZERO]; // main[0]=7 in current row
        let alpha = vec![QuadFelt::ONE];
        let inputs = RowInputs {
            main_pair: &main_pair,
            aux_pair: &[],
            periodic: &[],
            public_values: &[],
            randomness: &[],
            permutation_values: &[],
            is_first_row: Felt::ZERO,
            is_last_row: Felt::ZERO,
            is_transition: Felt::ZERO,
            alpha_powers_global: &alpha,
        };
        // Expected: 1 * 7 = 7 (lifted to QuadFelt).
        assert_eq!(run_tape(&tape, &inputs), QuadFelt::from(Felt::from_u32(7)));
    }

    /// Tape: a*b - c base constraint. main row = [a, b, c]. Should be 0
    /// when a*b = c.
    #[test]
    fn ab_minus_c_constraint() {
        let tape = AirTape {
            instructions: vec![
                Instruction::new(OP_LOAD_MAIN, 0, 0, 0), // r0 = main[0]
                Instruction::new(OP_LOAD_MAIN, 1, 0, 1), // r1 = main[1]
                Instruction::new(OP_LOAD_MAIN, 2, 0, 2), // r2 = main[2]
                Instruction::new(OP_MUL_BASE, 0, 1, 3),  // r3 = r0 * r1
                Instruction::new(OP_SUB_BASE, 3, 2, 4),  // r4 = r3 - r2
                Instruction::new(OP_ASSERT_ZERO_BASE, 4, 0, 0), // acc += alpha[0] * r4
            ],
            inline_consts: Vec::new(),
            base_reg_count: 5,
            ext_reg_count: 0,
            constraint_count: 1,
        };
        let alpha = vec![QuadFelt::ONE];
        let main_pair = vec![
            Felt::from_u32(3),
            Felt::from_u32(5),
            Felt::from_u32(15), // 3 * 5 = 15; constraint = 0
            Felt::ZERO,
            Felt::ZERO,
            Felt::ZERO,
        ];
        let inputs = RowInputs {
            main_pair: &main_pair,
            aux_pair: &[],
            periodic: &[],
            public_values: &[],
            randomness: &[],
            permutation_values: &[],
            is_first_row: Felt::ZERO,
            is_last_row: Felt::ZERO,
            is_transition: Felt::ZERO,
            alpha_powers_global: &alpha,
        };
        assert_eq!(run_tape(&tape, &inputs), QuadFelt::ZERO);

        // Off-by-one: change c so a*b ≠ c.
        let main_pair_bad = vec![
            Felt::from_u32(3),
            Felt::from_u32(5),
            Felt::from_u32(14), // 3*5 - 14 = 1 → acc = 1
            Felt::ZERO,
            Felt::ZERO,
            Felt::ZERO,
        ];
        let inputs_bad = RowInputs {
            main_pair: &main_pair_bad,
            ..inputs
        };
        assert_eq!(
            run_tape(&tape, &inputs_bad),
            QuadFelt::from(Felt::from_u32(1))
        );
    }

    /// Ext constraint: aux[0] * aux[1] - aux[2]. With aux[0]=2, aux[1]=3,
    /// aux[2]=6 → 0; with aux[2]=5 → 1.
    #[test]
    fn ext_ab_minus_c_constraint() {
        let tape = AirTape {
            instructions: vec![
                Instruction::new(OP_LOAD_AUX, 0, 0, 0), // e0 = aux[0]
                Instruction::new(OP_LOAD_AUX, 1, 0, 1), // e1 = aux[1]
                Instruction::new(OP_LOAD_AUX, 2, 0, 2), // e2 = aux[2]
                Instruction::new(OP_MUL_EXT, 0, 1, 3),  // e3 = e0*e1
                Instruction::new(OP_SUB_EXT, 3, 2, 4),  // e4 = e3 - e2
                Instruction::new(OP_ASSERT_ZERO_EXT, 4, 0, 0), // acc += alpha[0] * e4
            ],
            inline_consts: Vec::new(),
            base_reg_count: 0,
            ext_reg_count: 5,
            constraint_count: 1,
        };
        let alpha = vec![QuadFelt::ONE];
        let q = |x| QuadFelt::from(Felt::from_u32(x));
        let aux_pair = vec![q(2), q(3), q(6), q(0), q(0), q(0)];
        let inputs = RowInputs {
            main_pair: &[],
            aux_pair: &aux_pair,
            periodic: &[],
            public_values: &[],
            randomness: &[],
            permutation_values: &[],
            is_first_row: Felt::ZERO,
            is_last_row: Felt::ZERO,
            is_transition: Felt::ZERO,
            alpha_powers_global: &alpha,
        };
        assert_eq!(run_tape(&tape, &inputs), QuadFelt::ZERO);

        let aux_pair_bad = vec![q(2), q(3), q(5), q(0), q(0), q(0)];
        let inputs_bad = RowInputs {
            aux_pair: &aux_pair_bad,
            ..inputs
        };
        assert_eq!(run_tape(&tape, &inputs_bad), q(1));
    }
}
