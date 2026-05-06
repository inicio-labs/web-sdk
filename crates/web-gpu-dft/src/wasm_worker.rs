//! GPU worker entry point. Runs inside a dedicated Web Worker that owns the
//! `wgpu::Device` and dispatches GPU work in response to commands posted via
//! the SharedArrayBuffer.
//!
//! The worker uses `Atomics.waitAsync` (NOT the blocking `wait`) so it can
//! continue driving async wgpu operations between commands. The prover-side
//! client uses the blocking `Atomics::wait` to suspend the prover thread
//! until the worker writes the result.

#![cfg(all(feature = "real-gpu", target_arch = "wasm32"))]

use alloc::format;
use alloc::vec::Vec;

use js_sys::{Atomics, Int32Array, Promise, Reflect, SharedArrayBuffer, Uint8Array};
use miden_crypto::Felt;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::WgpuContext;
use crate::sab::*;

/// Async loop run inside the GPU worker. Acquires a `WgpuContext` and then
/// loops forever processing commands posted to the SAB.
///
/// Returns only on init failure (no GPU adapter).
#[wasm_bindgen(js_name = "runGpuWorker")]
pub async fn run_gpu_worker(sab: SharedArrayBuffer) -> Result<(), JsValue> {
    let ctx = WgpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU init failed: {e}")))?;

    let header = Int32Array::new(&sab);

    // Signal "ready" via console — the worker JS shim relays this back to the
    // main worker via postMessage in its outer onmessage handler.
    web_sys::console::log_1(&"[gpu-worker] adapter + device ready, entering command loop".into());

    loop {
        // Wait until cmd_signal becomes READY. Atomics::wait_async returns
        // `{async: bool, value: Promise|string}`.
        let r_obj = Atomics::wait_async(&header, CMD_SIGNAL_IDX as u32, SIGNAL_IDLE)?;
        let r: JsValue = r_obj.into();
        let r_async = Reflect::get(&r, &"async".into())?.as_bool().unwrap_or(false);
        let r_value = Reflect::get(&r, &"value".into())?;
        if r_async {
            let promise: Promise = r_value.dyn_into()?;
            JsFuture::from(promise).await?;
        }
        // else: value is "not-equal" (signal already changed) or "timed-out"
        // (we passed no timeout so this doesn't happen). Either way, proceed
        // to process whatever the current cmd is.

        // Re-read the signal to make sure we actually have a cmd. If for
        // some reason we got woken without READY (spurious notify), loop.
        let cur = Atomics::load(&header, CMD_SIGNAL_IDX as u32)?;
        if cur != SIGNAL_READY {
            continue;
        }

        // Decode header.
        let op = Atomics::load(&header, CMD_OP_IDX as u32)? as u32;
        let rows = Atomics::load(&header, ROWS_IDX as u32)? as usize;
        let cols = Atomics::load(&header, COLS_IDX as u32)? as usize;
        let added_bits = Atomics::load(&header, ADDED_BITS_IDX as u32)? as usize;
        let shift_lo = Atomics::load(&header, SHIFT_LO_IDX as u32)? as u32;
        let shift_hi = Atomics::load(&header, SHIFT_HI_IDX as u32)? as u32;
        let input_byte_len = Atomics::load(&header, INPUT_LEN_IDX as u32)? as usize;
        let shift = Felt::new((shift_lo as u64) | ((shift_hi as u64) << 32));

        // Copy input from payload into a Rust Vec<u8>, then decode to Felts.
        let mut input_bytes: Vec<u8> = alloc::vec![0u8; input_byte_len];
        let in_view = Uint8Array::new_with_byte_offset_and_length(
            &sab,
            PAYLOAD_OFFSET as u32,
            input_byte_len as u32,
        );
        in_view.copy_to(&mut input_bytes);
        let mut input: Vec<Felt> = Vec::with_capacity(input_byte_len / 8);
        for chunk in input_bytes.chunks_exact(8) {
            let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            input.push(Felt::new((lo as u64) | ((hi as u64) << 32)));
        }

        // Run the requested op. Errors are caught and signalled via error_flag;
        // the client returns zeros in that case.
        let output: Vec<Felt> = match op {
            x if x == OP_DFT_BATCH => ctx.gl_dft_batch_async(&input, rows, cols).await,
            x if x == OP_IDFT_BATCH => ctx.gl_idft_batch_async(&input, rows, cols).await,
            x if x == OP_COSET_LDE_BATCH => {
                ctx.gl_coset_lde_batch_async(&input, rows, cols, added_bits, shift).await
            },
            _ => {
                Atomics::store(&header, ERROR_FLAG_IDX as u32, 1)?;
                Vec::new()
            },
        };

        // Encode output back into payload area as (lo, hi) u32 pairs.
        let mut packed: Vec<u8> = Vec::with_capacity(output.len() * 8);
        for f in &output {
            let v = f.as_canonical_u64();
            packed.extend_from_slice(&(v as u32).to_le_bytes());
            packed.extend_from_slice(&((v >> 32) as u32).to_le_bytes());
        }
        if !packed.is_empty() {
            let out_view = Uint8Array::new_with_length(packed.len() as u32);
            out_view.copy_from(&packed);
            let payload_view = Uint8Array::new_with_byte_offset_and_length(
                &sab,
                PAYLOAD_OFFSET as u32,
                packed.len() as u32,
            );
            payload_view.set(&out_view, 0);
        }
        Atomics::store(&header, OUTPUT_LEN_IDX as u32, packed.len() as i32)?;

        // Reset cmd_signal to IDLE, set result_signal to READY, notify.
        Atomics::store(&header, CMD_SIGNAL_IDX as u32, SIGNAL_IDLE)?;
        Atomics::store(&header, RESULT_SIGNAL_IDX as u32, SIGNAL_READY)?;
        Atomics::notify(&header, RESULT_SIGNAL_IDX as u32)?;
    }
}
