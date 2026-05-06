//! Synchronous client used by the prover thread to dispatch GPU work to the
//! GPU worker via SharedArrayBuffer + Atomics.
//!
//! Each operation:
//!   1. Write input bytes + header params into the SAB payload area
//!   2. Set `cmd_op`, `cmd_signal = READY`, `result_signal = IDLE`
//!   3. `Atomics::notify(cmd_signal)`  — wakes the worker
//!   4. `Atomics::wait(result_signal, IDLE)`  — BLOCKS the prover thread
//!      until the worker writes `result_signal = READY`
//!   5. Read output bytes from SAB payload, decode to `Vec<Felt>`
//!
//! `Atomics::wait` only works on a Web Worker, never the main thread. The
//! SDK's prove path runs inside `web-client-methods-worker` which IS a worker,
//! so this is fine. (If newGpuProver were ever called from the main thread,
//! the wait would throw.)

#![cfg(all(feature = "real-gpu", target_arch = "wasm32"))]

use alloc::vec::Vec;

use js_sys::{Atomics, Int32Array, SharedArrayBuffer, Uint8Array};
use miden_crypto::Felt;
use p3_field::PrimeCharacteristicRing;

use crate::sab::*;

/// Sync GPU client. Holds the SAB shared with the GPU worker. Cheap to clone.
#[derive(Clone)]
pub struct GpuClient {
    sab: SharedArrayBuffer,
}

// Safety: SharedArrayBuffer is thread-shareable by construction (that's its
// point). wasm-bindgen marks JsValue wrappers as !Send by default; we override
// because the SAB IS specifically meant to cross worker boundaries.
unsafe impl Send for GpuClient {}
unsafe impl Sync for GpuClient {}

impl GpuClient {
    pub fn new(sab: SharedArrayBuffer) -> Self {
        Self { sab }
    }

    pub fn dft_batch(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        self.run_op(OP_DFT_BATCH, data, rows, cols, 0, Felt::ZERO, rows * cols)
    }

    pub fn idft_batch(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        self.run_op(OP_IDFT_BATCH, data, rows, cols, 0, Felt::ZERO, rows * cols)
    }

    pub fn coset_lde_batch(
        &self,
        data: &[Felt],
        rows: usize,
        cols: usize,
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        let output_felts = (rows << added_bits) * cols;
        self.run_op(OP_COSET_LDE_BATCH, data, rows, cols, added_bits, shift, output_felts)
    }

    fn run_op(
        &self,
        op: u32,
        data: &[Felt],
        rows: usize,
        cols: usize,
        added_bits: usize,
        shift: Felt,
        output_felts: usize,
    ) -> Vec<Felt> {
        // Header view: i32 array starting at byte 0. Atomics use i32 indices.
        let header_i32 = Int32Array::new(&self.sab);
        let payload_u8 = Uint8Array::new_with_byte_offset_and_length(
            &self.sab,
            PAYLOAD_OFFSET as u32,
            PAYLOAD_SIZE as u32,
        );

        // Pack input as [lo, hi] u32 pairs (little-endian byte order, matches
        // bytemuck::cast_slice on the worker side).
        let input_byte_len = data.len() * 8;
        let mut packed: Vec<u8> = Vec::with_capacity(input_byte_len);
        for f in data {
            let v = f.as_canonical_u64();
            packed.extend_from_slice(&(v as u32).to_le_bytes());
            packed.extend_from_slice(&((v >> 32) as u32).to_le_bytes());
        }
        let input_view = Uint8Array::new_with_length(packed.len() as u32);
        input_view.copy_from(&packed);
        payload_u8.set(&input_view, 0);

        // Write header params via plain stores (no atomics needed for non-signal fields).
        header_i32.set_index(CMD_OP_IDX as u32, op as i32);
        header_i32.set_index(ERROR_FLAG_IDX as u32, 0);
        header_i32.set_index(ROWS_IDX as u32, rows as i32);
        header_i32.set_index(COLS_IDX as u32, cols as i32);
        header_i32.set_index(ADDED_BITS_IDX as u32, added_bits as i32);
        let shift_u64 = shift.as_canonical_u64();
        header_i32.set_index(SHIFT_LO_IDX as u32, shift_u64 as i32);
        header_i32.set_index(SHIFT_HI_IDX as u32, (shift_u64 >> 32) as i32);
        header_i32.set_index(INPUT_LEN_IDX as u32, input_byte_len as i32);
        header_i32.set_index(OUTPUT_LEN_IDX as u32, 0);

        // Reset result signal, then set cmd signal and notify.
        Atomics::store(&header_i32, RESULT_SIGNAL_IDX as u32, SIGNAL_IDLE).unwrap();
        Atomics::store(&header_i32, CMD_SIGNAL_IDX as u32, SIGNAL_READY).unwrap();
        Atomics::notify(&header_i32, CMD_SIGNAL_IDX as u32).unwrap();

        // Block prover thread until worker sets result_signal to READY.
        // Atomics::wait blocks if the current value at the index equals the expected
        // value (here SIGNAL_IDLE); when notified or the value changes, it returns.
        let _ = Atomics::wait(&header_i32, RESULT_SIGNAL_IDX as u32, SIGNAL_IDLE);

        // Check error flag.
        let error = header_i32.get_index(ERROR_FLAG_IDX as u32);
        if error != 0 {
            // Worker hit a kernel/dispatch error. Return zeros for now; a future
            // revision would propagate an explicit Result.
            return alloc::vec![Felt::ZERO; output_felts];
        }

        // Read output bytes from payload area.
        let output_byte_len = (output_felts * 8) as usize;
        let mut out_bytes: Vec<u8> = alloc::vec![0u8; output_byte_len];
        let out_view = Uint8Array::new_with_byte_offset_and_length(
            &self.sab,
            PAYLOAD_OFFSET as u32,
            output_byte_len as u32,
        );
        out_view.copy_to(&mut out_bytes);

        // Decode (lo, hi) u32 pairs back to Felt.
        let mut out: Vec<Felt> = Vec::with_capacity(output_felts);
        for chunk in out_bytes.chunks_exact(8) {
            let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let v = (lo as u64) | ((hi as u64) << 32);
            out.push(Felt::new(v));
        }

        // Reset cmd signal so the worker can wait_async on the next iteration.
        Atomics::store(&header_i32, CMD_SIGNAL_IDX as u32, SIGNAL_IDLE).unwrap();

        out
    }
}
