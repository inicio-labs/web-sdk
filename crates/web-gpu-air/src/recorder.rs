//! Runs `<ProcessorAir as Air>::eval` against a `SymbolicAirBuilder` and exposes
//! the recorded constraint DAGs + layout.
//!
//! Phase 0a Unit 2 (Risk c): verify that `MainTraceRow<T>::borrow_from_slice`
//! works when `T = SymbolicVariable<Felt>`. The CPU prover uses
//! `T = AB::Var = Felt` (8 bytes, alignment 8); the symbolic recorder needs
//! `T = SymbolicVariable<Felt>` (entry: BaseEntry + index: usize + PhantomData =
//! 24 bytes, alignment 8). Field count and alignment match, so the unsafe
//! `align_to::<MainTraceRow<T>>` cast in the Borrow impl
//! (`miden-vm/air/src/trace/main_trace.rs:68-77`) should produce exactly one
//! item with no prefix/suffix. Failure of any of its four `debug_assert!`s
//! invalidates the SymbolicAirBuilder approach for the Miden recorder.

#[cfg(test)]
mod risk_c_alignment {
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

    /// Document the byte layout we just verified, for the next reader.
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
