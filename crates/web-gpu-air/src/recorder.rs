//! Runs `<ProcessorAir as Air>::eval` against a `SymbolicAirBuilder` and exposes
//! the recorded constraint DAGs + layout.

use miden_air::{
    NUM_PUBLIC_VALUES, ProcessorAir,
    trace::{AUX_TRACE_RAND_CHALLENGES, AUX_TRACE_WIDTH, TRACE_WIDTH},
};
use miden_crypto::{
    Felt,
    field::BinomialExtensionField,
    stark::air::LiftedAir,
};
use p3_air::{
    AirLayout, ConstraintLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
};

/// Quadratic extension over Goldilocks (W=7), the same EF the prover uses.
pub type QuadFelt = BinomialExtensionField<Felt, 2>;

/// One-shot recording of Miden's `ProcessorAir`: base/ext constraint DAGs
/// plus the layout tagging each global constraint as base or ext.
#[derive(Debug)]
pub struct RecordedAir {
    pub layout: AirLayout,
    pub constraint_layout: ConstraintLayout,
    pub constraints_base: Vec<SymbolicExpression<Felt>>,
    pub constraints_ext: Vec<SymbolicExpressionExt<Felt, QuadFelt>>,
}

/// Build the `AirLayout` matching Miden's `ProcessorAir`. All seven fields
/// of `AirLayout` are populated; the values are pulled from miden-air constants
/// where possible and from `<ProcessorAir as LiftedAir>` accessors otherwise.
pub fn miden_air_layout() -> AirLayout {
    let air = ProcessorAir;
    let num_periodic_columns = <ProcessorAir as LiftedAir<Felt, QuadFelt>>::periodic_columns(&air)
        .len();
    AirLayout {
        // Miden has no preprocessed columns.
        preprocessed_width: 0,
        main_width: TRACE_WIDTH,
        num_public_values: NUM_PUBLIC_VALUES,
        // SymbolicAirBuilder treats permutation as RowMajorMatrix<SymbolicVariableExt>,
        // one variable per EF column — no doubling. AUX_TRACE_WIDTH = 8 (EF cols).
        permutation_width: AUX_TRACE_WIDTH,
        num_permutation_challenges: AUX_TRACE_RAND_CHALLENGES,
        // num_aux_values() == AUX_TRACE_WIDTH for Miden.
        num_permutation_values: AUX_TRACE_WIDTH,
        num_periodic_columns,
    }
}

/// Run `<ProcessorAir as Air<SymbolicAirBuilder>>::eval` once and capture the
/// recorded constraint DAGs and layout.
///
/// Builds the SymbolicAirBuilder explicitly (not via the
/// `get_all_symbolic_constraints` helper) so we can grab `constraint_layout()`
/// and the constraint vectors from the same instance.
pub fn record_processor_air() -> RecordedAir {
    let layout = miden_air_layout();
    let mut builder = SymbolicAirBuilder::<Felt, QuadFelt>::new(layout);
    <ProcessorAir as LiftedAir<Felt, QuadFelt>>::eval(&ProcessorAir, &mut builder);
    let constraint_layout = builder.constraint_layout();
    let constraints_base = builder.base_constraints();
    let constraints_ext = builder.extension_constraints();
    RecordedAir {
        layout,
        constraint_layout,
        constraints_base,
        constraints_ext,
    }
}

#[cfg(test)]
mod risk_c_alignment {
    //! Phase 0a Unit 2 (Risk c): verify that `MainTraceRow<T>::borrow` works
    //! when `T = SymbolicVariable<Felt>`. The CPU prover uses
    //! `T = AB::Var = Felt`; the symbolic recorder needs
    //! `T = SymbolicVariable<Felt>`. Field count and alignment must match for
    //! the unsafe `align_to::<MainTraceRow<T>>` cast at
    //! `miden-vm/air/src/trace/main_trace.rs:68-77` to produce exactly one item
    //! with no prefix/suffix. Failure of any of its four `debug_assert!`s
    //! would invalidate the SymbolicAirBuilder approach for the Miden recorder.

    use core::borrow::Borrow;

    use miden_air::trace::{MainTraceRow, TRACE_WIDTH};
    use miden_crypto::Felt;
    use p3_air::{BaseEntry, SymbolicVariable};

    /// Construct a slice of `TRACE_WIDTH` `SymbolicVariable<Felt>`s — exactly
    /// what `SymbolicAirBuilder::main` produces per row — and try to borrow it
    /// as `&MainTraceRow<SymbolicVariable<Felt>>`. Trips the Borrow impl's
    /// debug_asserts on misalignment / size mismatch / empty `shorts`.
    #[test]
    fn main_trace_row_borrow_with_symbolic_variable() {
        // Build the slice the way SymbolicAirBuilder::new does at builder.rs:167-173:
        // for each main column index, emit a SymbolicVariable with
        // BaseEntry::Main { offset: 0 }.
        let row: Vec<SymbolicVariable<Felt>> = (0..TRACE_WIDTH)
            .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index))
            .collect();

        assert_eq!(row.len(), TRACE_WIDTH, "TRACE_WIDTH mismatch");

        // The Borrow impl in miden-vm/air/src/trace/main_trace.rs uses
        // `slice.align_to::<MainTraceRow<T>>` under the hood. In dev/test
        // builds (debug_assertions on), the four asserts inside fire on any
        // alignment / size mismatch. Just calling .borrow() is the test.
        let view: &MainTraceRow<SymbolicVariable<Felt>> = row.as_slice().borrow();

        // Sanity: read back the column indices via the typed-field view.
        // - clk should be column 0 (first MainTraceRow field).
        // - ctx should be column 1.
        assert_eq!(
            view.clk.index, 0,
            "clk column index must be 0 (first field of MainTraceRow)"
        );
        assert_eq!(
            view.ctx.index, 1,
            "ctx column index must be 1 (second field of MainTraceRow)"
        );
        // fn_hash[0..4] are columns 2..6.
        for i in 0..4 {
            assert_eq!(
                view.fn_hash[i].index,
                2 + i,
                "fn_hash[{i}] column index mismatch"
            );
        }

        // Size sanity: the type-punned cast must produce exactly one
        // MainTraceRow's worth of items.
        assert_eq!(
            core::mem::size_of_val(view),
            core::mem::size_of::<SymbolicVariable<Felt>>() * TRACE_WIDTH,
            "MainTraceRow<SymbolicVariable<Felt>> size != T size * TRACE_WIDTH"
        );

        // Alignment sanity.
        assert_eq!(
            core::mem::align_of::<MainTraceRow<SymbolicVariable<Felt>>>(),
            core::mem::align_of::<SymbolicVariable<Felt>>(),
            "MainTraceRow alignment must match SymbolicVariable alignment"
        );

        // If we got here without panicking, the `align_to`-based cast works
        // and the column-index ordering matches MainTraceRow's field order.
        // Phase 0a Risk (c) PASSES.
    }

    #[test]
    fn document_layout() {
        let sv_size = core::mem::size_of::<SymbolicVariable<Felt>>();
        let sv_align = core::mem::align_of::<SymbolicVariable<Felt>>();
        let row_size = core::mem::size_of::<MainTraceRow<SymbolicVariable<Felt>>>();
        let row_align = core::mem::align_of::<MainTraceRow<SymbolicVariable<Felt>>>();
        eprintln!(
            "SymbolicVariable<Felt>: size={sv_size} align={sv_align}\n\
             MainTraceRow<SymbolicVariable<Felt>>: size={row_size} align={row_align}\n\
             TRACE_WIDTH={TRACE_WIDTH}\n\
             Expected row_size = {} (= sv_size * TRACE_WIDTH)",
            sv_size * TRACE_WIDTH,
        );
        assert_eq!(row_size, sv_size * TRACE_WIDTH);
    }
}

#[cfg(test)]
mod risk_b_smoke {
    //! Phase 0a Unit 3 (Risk b): instantiate `SymbolicAirBuilder<Felt, QuadFelt>`
    //! with the full Miden `AirLayout`, run `<ProcessorAir as Air>::eval` against
    //! it, confirm we get non-empty constraint vectors and a coherent layout.
    //! No new impls should be needed — `TaggingAirBuilderExt` chains via blanket
    //! impls (`miden-vm/air/src/constraints/tagging/{enabled,fallback}.rs`).

    use super::*;

    #[test]
    fn miden_air_layout_fields() {
        let layout = miden_air_layout();
        assert_eq!(layout.preprocessed_width, 0, "Miden has no preprocessed cols");
        assert_eq!(layout.main_width, TRACE_WIDTH);
        assert_eq!(layout.main_width, 71, "TRACE_WIDTH spec'd as 71");
        assert_eq!(layout.num_public_values, NUM_PUBLIC_VALUES);
        assert_eq!(layout.permutation_width, AUX_TRACE_WIDTH);
        assert_eq!(layout.permutation_width, 8, "AUX_TRACE_WIDTH spec'd as 8");
        assert_eq!(layout.num_permutation_challenges, AUX_TRACE_RAND_CHALLENGES);
        assert_eq!(layout.num_permutation_challenges, 2);
        assert_eq!(layout.num_permutation_values, AUX_TRACE_WIDTH);
        assert!(layout.num_periodic_columns > 0, "expect ≥ 1 periodic column from chiplets");
        eprintln!("AirLayout: {layout:?}");
    }

    #[test]
    fn record_processor_air_smoke() {
        let recorded = record_processor_air();

        let n_base = recorded.constraints_base.len();
        let n_ext = recorded.constraints_ext.len();
        let n_total = recorded.constraint_layout.base_indices.len()
            + recorded.constraint_layout.ext_indices.len();

        eprintln!(
            "ProcessorAir recorded: base_constraints={n_base} ext_constraints={n_ext} \
             total_via_layout={n_total} periodic_cols={}",
            recorded.layout.num_periodic_columns
        );

        // Sanity: constraint vectors are non-empty (Miden has hundreds of constraints).
        assert!(n_base > 0, "expected non-empty base constraint vector");
        assert!(n_ext > 0, "expected non-empty ext constraint vector (LogUp bus)");

        // ConstraintLayout's bucket sizes must match the actual vector sizes.
        assert_eq!(
            recorded.constraint_layout.base_indices.len(),
            n_base,
            "ConstraintLayout.base_indices size mismatch"
        );
        assert_eq!(
            recorded.constraint_layout.ext_indices.len(),
            n_ext,
            "ConstraintLayout.ext_indices size mismatch"
        );

        // Every global index 0..n_total appears exactly once across the two buckets.
        let mut seen = vec![false; n_total];
        for &i in &recorded.constraint_layout.base_indices {
            assert!(i < n_total, "base index {i} out of bounds");
            assert!(!seen[i], "duplicate global index {i} in base bucket");
            seen[i] = true;
        }
        for &i in &recorded.constraint_layout.ext_indices {
            assert!(i < n_total, "ext index {i} out of bounds");
            assert!(!seen[i], "duplicate global index {i} in ext bucket");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "ConstraintLayout missed at least one global index");
    }
}
