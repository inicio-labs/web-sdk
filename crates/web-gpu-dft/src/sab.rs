//! SharedArrayBuffer protocol between the prover thread and the GPU worker.
//!
//! Layout (all offsets in bytes):
//!
//! ```text
//!   0..4    cmd_signal     : i32  — 0 idle, 1 cmd ready (Atomics.wait/notify target)
//!   4..8    result_signal  : i32  — 0 idle, 1 result ready (Atomics.wait/notify target)
//!   8..12   cmd_op         : u32  — 0=dft_batch, 1=idft_batch, 2=coset_lde_batch
//!  12..16   error_flag     : u32  — 0=ok, 1=error (worker sets if dispatch failed)
//!  16..20   rows           : u32
//!  20..24   cols           : u32
//!  24..28   added_bits     : u32  (only used by coset_lde)
//!  28..32   shift_lo       : u32  (only used by coset_lde)
//!  32..36   shift_hi       : u32  (only used by coset_lde)
//!  36..40   input_byte_len : u32  — set by client
//!  40..44   output_byte_len: u32  — set by worker
//!  44..64   reserved
//!  64..N    payload area   — input then output, packed Felts as u32 little-endian pairs
//! ```
//!
//! The payload area is sized at SAB construction (currently 256 MiB) to fit the
//! largest expected matrix `rows × cols × 8 bytes` plus the post-LDE blowup.
//! Input and output share the same payload area: writer writes input at offset 0,
//! reader writes output at the same offset (overwriting input, since input is
//! consumed first).

#![cfg(feature = "real-gpu")]

/// Header byte size, where the payload starts.
pub const PAYLOAD_OFFSET: usize = 64;

/// Payload area size. The Miden VM's TRACE_WIDTH is 72 cols at 8 bytes each,
/// and the prover passes post-LDE matrices (rows up to ~2^20 = 1M) through
/// dft_batch — that's ~600 MiB for a single dft call. coset_lde_batch's
/// input is the pre-LDE trace (~75 MiB), but its output is 8× larger.
/// 1.5 GiB lets us handle ~700 MiB inputs which is the practical ceiling
/// for dft_batch and the 8×-blowup output of coset_lde on ~175 MiB inputs.
/// SharedArrayBuffer max is ~2 GiB on Chrome desktop; we leave headroom for
/// both pages of the buffer plus the OS allocator's overhead.
pub const PAYLOAD_SIZE: usize = 1536 * 1024 * 1024;

/// Total SAB size.
pub const SAB_SIZE: usize = PAYLOAD_OFFSET + PAYLOAD_SIZE;

// Header offsets, in i32/u32 indices (each 4 bytes).
pub const CMD_SIGNAL_IDX: usize     = 0;  // i32 at byte 0
pub const RESULT_SIGNAL_IDX: usize  = 1;  // i32 at byte 4
pub const CMD_OP_IDX: usize         = 2;  // u32 at byte 8
pub const ERROR_FLAG_IDX: usize     = 3;  // u32 at byte 12
pub const ROWS_IDX: usize           = 4;  // u32 at byte 16
pub const COLS_IDX: usize           = 5;  // u32 at byte 20
pub const ADDED_BITS_IDX: usize     = 6;  // u32 at byte 24
pub const SHIFT_LO_IDX: usize       = 7;  // u32 at byte 28
pub const SHIFT_HI_IDX: usize       = 8;  // u32 at byte 32
pub const INPUT_LEN_IDX: usize      = 9;  // u32 at byte 36
pub const OUTPUT_LEN_IDX: usize     = 10; // u32 at byte 40

// Op codes.
pub const OP_DFT_BATCH: u32       = 0;
pub const OP_IDFT_BATCH: u32      = 1;
pub const OP_COSET_LDE_BATCH: u32 = 2;

// Signal values.
pub const SIGNAL_IDLE: i32 = 0;
pub const SIGNAL_READY: i32 = 1;
