//! `wgpu` device + queue acquisition and a buffer-roundtrip smoke test.
//!
//! Behind `cfg(feature = "real-gpu")`. When the feature is off, the crate
//! ships the CPU-delegating stub in `dft.rs` and this module is skipped.
//!
//! On wasm32 we use the `webgpu` backend (browser-side `navigator.gpu`).
//! On native we use whichever backend wgpu picks by default — Metal on
//! macOS, Vulkan / llvmpipe on Linux. The shaders are identical (WGSL is
//! cross-backend), so `cargo test --features real-gpu` on a dev machine
//! validates the same kernels that ship to browsers.

use alloc::borrow::ToOwned;
use alloc::string::ToString;
use alloc::vec::Vec;

use miden_crypto::Felt;
use p3_field::{Field, TwoAdicField};

use crate::GpuInitError;

/// Goldilocks WGSL primitives (mul_u32, mul_u64_to_u128, gl_reduce_u128, gl_add, gl_mul).
/// Concatenated into the front of every kernel that does Goldilocks math.
const GOLDILOCKS_WGSL: &str = include_str!("shaders/goldilocks.wgsl");

/// Powers of the primitive `n`-th root of unity: `[w_n^0, w_n^1, ..., w_n^{n/2-1}]`.
fn compute_twiddles(n: usize) -> Vec<Felt> {
    let log_n = n.trailing_zeros() as usize;
    let w_n = Felt::two_adic_generator(log_n);
    let mut out = Vec::with_capacity(n / 2);
    let mut acc = Felt::ONE;
    for _ in 0..n / 2 {
        out.push(acc);
        acc *= w_n;
    }
    out
}

/// Decode a `(lo, hi)` u32-pair byte buffer back into `Vec<Felt>`.
fn decode_packed_felts(bytes: Vec<u8>, expected_felts: usize) -> Vec<Felt> {
    let raw: &[u32] = bytemuck::cast_slice(&bytes);
    let mut out = Vec::with_capacity(expected_felts);
    for chunk in raw.chunks_exact(2).take(expected_felts) {
        let v = (chunk[0] as u64) | ((chunk[1] as u64) << 32);
        out.push(Felt::new(v));
    }
    out
}

/// Bit-reverse the low `bits` bits of `x`. Used CPU-side to permute between
/// natural and bit-reversed indexing.
fn reverse_bits_len(mut x: usize, bits: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Cross-target async readback: maps `staging` for read, drives the device
/// to completion (sync poll on native; the JS event loop on wasm32), and
/// returns the bytes. Wraps wgpu's callback-based `map_async` into a Future
/// via futures-channel::oneshot.
async fn read_staging_async(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
) -> Vec<u8> {
    let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    // Native: drive the queue to completion. The map_async callback fires
    // synchronously inside `poll(Wait)` so the channel is filled by the time
    // we await it. Wasm32: device.poll is a no-op; the JS event loop drives
    // mapAsync's Promise resolution, and rx.await yields until then.
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .expect("device poll");
    #[cfg(target_arch = "wasm32")]
    let _ = device; // silence unused on wasm32

    rx.await
        .expect("map_async oneshot dropped")
        .expect("map_async failed");

    let data = staging.slice(..).get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    staging.unmap();
    bytes
}

/// Owns a wgpu device + queue plus pre-compiled compute pipelines for the
/// hot kernels: multi-column NTT stage, divide-by-n, row-swap permutation,
/// coset shift. Each is dispatched in 2D with gid.y = column index, so a
/// single kernel call processes all columns of a row-major matrix at once.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Multi-column NTT/DIF butterfly stage. Used by both forward and inverse
    /// (inverse just runs forward with different post-processing).
    pub ntt_pipeline: wgpu::ComputePipeline,
    pub ntt_bind_layout: wgpu::BindGroupLayout,
    /// Multi-column post-DFT correction for iDFT: scale every row by n^{-1}
    /// then bit-reverse (we run forward DIF, output is bit-reversed; iDFT
    /// needs natural-order with the j → (n-j)%n permutation, which combined
    /// with the bit-reverse becomes a single index mapping kernel).
    pub idft_finalize_pipeline: wgpu::ComputePipeline,
    pub idft_finalize_bind_layout: wgpu::BindGroupLayout,
    /// Multi-column coset shift: scale row r by shift^r. Used by coset_lde
    /// after iDFT and zero-padding.
    pub coset_shift_pipeline: wgpu::ComputePipeline,
    pub coset_shift_bind_layout: wgpu::BindGroupLayout,
}

const NTT_KERNEL_SOURCE_TEMPLATE: &str = r#"
{prelude}

struct NttParams {
    half     : u32,
    log_step : u32,
    n_half   : u32,
    cols     : u32,
};

@group(0) @binding(0) var<storage, read_write> data     : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       twiddles : array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params   : NttParams;

// 2D dispatch:
//   gid.x in [0, n_half) — butterfly index within a column
//   gid.y in [0, cols)   — column index
// Memory layout: data[row * cols + col] (row-major).
@compute @workgroup_size(64, 1, 1)
fn ntt_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let butterfly = gid.x;
    let col = gid.y;
    if (butterfly >= params.n_half) { return; }
    if (col >= params.cols) { return; }

    let half = params.half;
    let group_size = half * 2u;
    let group = butterfly / half;
    let j     = butterfly % half;
    let i_a   = group * group_size + j;
    let i_b   = i_a + half;

    let idx_a = i_a * params.cols + col;
    let idx_b = i_b * params.cols + col;

    let a_val = data[idx_a];
    let b_val = data[idx_b];

    let stride = 1u << params.log_step;
    let tw_idx = j * stride;
    let tw     = twiddles[tw_idx];

    let sum   = gl_add(a_val, b_val);
    let diff  = gl_sub(a_val, b_val);
    let mul_v = gl_mul(diff, tw);

    data[idx_a] = sum;
    data[idx_b] = mul_v;
}
"#;

const IDFT_FINALIZE_TEMPLATE: &str = r#"
{prelude}

struct FinalizeParams {
    n        : u32,
    log_n    : u32,
    cols     : u32,
    n_inv_lo : u32,
    n_inv_hi : u32,
    pad0     : u32,
    pad1     : u32,
    pad2     : u32,
};

// Forward NTT produced bit-reversed-order data. Plonky3's iDFT default impl
// post-processes the natural-order DFT by:
//   1. bit-reverse (i.e., un-do the bit-reversal so we have natural order)
//   2. divide every element by n
//   3. swap rows 1..n/2 with rows n-1..n/2  (j -> (n-j) mod n permutation)
// We fuse all three into one kernel using the index transform:
//   nat_idx_for_row_r = if r == 0 { 0 } else { n - r }
// then storing src_bit_rev[r] -> dst[nat_idx_for_row_r] / n.
@group(0) @binding(0) var<storage, read>       src : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params : FinalizeParams;

fn rev_bits_u32(x: u32, log_n: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < log_n; i = i + 1u) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(64, 1, 1)
fn idft_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;        // destination row in natural order
    let col = gid.y;
    if (r >= params.n) { return; }
    if (col >= params.cols) { return; }

    // We want: dst[r, col] = src_natural[(n-r) mod n, col] / n
    //        = src_bit_reversed[bit_rev((n-r) mod n), col] / n
    var nat_idx: u32;
    if (r == 0u) { nat_idx = 0u; } else { nat_idx = params.n - r; }
    let src_idx = rev_bits_u32(nat_idx, params.log_n) * params.cols + col;

    let v = src[src_idx];
    let n_inv = vec2<u32>(params.n_inv_lo, params.n_inv_hi);
    let scaled = gl_mul(v, n_inv);

    dst[r * params.cols + col] = scaled;
}
"#;

const COSET_SHIFT_TEMPLATE: &str = r#"
{prelude}

struct CosetShiftParams {
    rows     : u32,
    cols     : u32,
    pad0     : u32,
    pad1     : u32,
};

// Multiplies row r by shift_powers[r], in place. shift_powers is a
// pre-uploaded array of [shift^0, shift^1, ..., shift^{rows-1}].
@group(0) @binding(0) var<storage, read_write> data         : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       shift_powers : array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params       : CosetShiftParams;

@compute @workgroup_size(64, 1, 1)
fn coset_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let col = gid.y;
    if (r >= params.rows) { return; }
    if (col >= params.cols) { return; }
    let idx = r * params.cols + col;
    data[idx] = gl_mul(data[idx], shift_powers[r]);
}
"#;

impl WgpuContext {
    /// Acquire a device + queue from the default adapter for the current
    /// platform. Async because wgpu's adapter / device acquisition is
    /// inherently async.
    pub async fn new() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuInitError::AdapterUnavailable)?;

        // Log which adapter we got so the consumer can confirm GPU vs software.
        #[cfg(target_arch = "wasm32")]
        {
            let info = adapter.get_info();
            web_sys::console::log_1(
                &alloc::format!(
                    "[gpu-worker] adapter: name={:?} vendor={} device={} backend={:?} type={:?}",
                    info.name,
                    info.vendor,
                    info.device,
                    info.backend,
                    info.device_type,
                )
                .into(),
            );
        }

        // Default WebGPU limits cap maxBufferSize at 256 MB and
        // maxStorageBufferBindingSize at 128 MB. Miden's post-LDE matrices
        // routinely exceed these (~600 MB for 65k×72 + 8× blowup). Negotiate
        // up to whatever the adapter permits.
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        // Compute pipelines that use these buffers also need to allow the
        // dispatch grids we're using — bump the workgroup-count limits too.
        required_limits.max_compute_workgroups_per_dimension =
            adapter_limits.max_compute_workgroups_per_dimension;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("miden-web-gpu-dft device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| GpuInitError::DeviceInit(e.to_string()))?;

        // Pre-compile the hot kernels once on init.
        let make_pipeline = |label: &str, src: alloc::string::String, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipeline.get_bind_group_layout(0);
            (pipeline, layout)
        };

        let (ntt_pipeline, ntt_bind_layout) = make_pipeline(
            "ntt_stage (multi-col)",
            NTT_KERNEL_SOURCE_TEMPLATE.replace("{prelude}", GOLDILOCKS_WGSL),
            "ntt_stage",
        );
        let (idft_finalize_pipeline, idft_finalize_bind_layout) = make_pipeline(
            "idft_finalize",
            IDFT_FINALIZE_TEMPLATE.replace("{prelude}", GOLDILOCKS_WGSL),
            "idft_finalize",
        );
        let (coset_shift_pipeline, coset_shift_bind_layout) = make_pipeline(
            "coset_shift",
            COSET_SHIFT_TEMPLATE.replace("{prelude}", GOLDILOCKS_WGSL),
            "coset_shift",
        );

        Ok(Self {
            device,
            queue,
            ntt_pipeline,
            ntt_bind_layout,
            idft_finalize_pipeline,
            idft_finalize_bind_layout,
            coset_shift_pipeline,
            coset_shift_bind_layout,
        })
    }

    /// Smoke test: round-trip a u32 buffer through the GPU via an identity
    /// compute kernel. Exercises every step of the wgpu pipeline (buffer
    /// upload, shader compile, dispatch, mapAsync readback). If this passes
    /// on a target, the toolchain is healthy enough to run real kernels.
    ///
    /// Native-target only — uses `device.poll(Wait)` to block until the
    /// readback completes. On wasm32 the equivalent is
    /// [`Self::buffer_roundtrip_u32_async`] which awaits a JS-side promise
    /// instead of blocking.
    pub async fn buffer_roundtrip_u32_async(&self, input: &[u32]) -> Vec<u32> {
        let (input_buf, output_buf, staging, pipeline, bind_group, bytes_u64) =
            self.identity_setup(input);
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("identity encoder") },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("identity pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = input.len().div_ceil(64) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, bytes_u64);
        self.queue.submit(Some(encoder.finish()));
        drop(input_buf);
        let bytes = read_staging_async(&self.device, &staging).await;
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Native sync wrapper: pollster::block_on the async core.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn buffer_roundtrip_u32(&self, input: &[u32]) -> Vec<u32> {
        pollster::block_on(self.buffer_roundtrip_u32_async(input))
    }

    /// Native sync wrapper: pollster::block_on the async core.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn elementwise_gl_mul(&self, a: &[Felt], b: &[Felt]) -> Vec<Felt> {
        pollster::block_on(self.elementwise_gl_mul_async(a, b))
    }

    /// Compute element-wise Goldilocks multiplication on the GPU. Async core
    /// works on both native and wasm32.
    pub async fn elementwise_gl_mul_async(&self, a: &[Felt], b: &[Felt]) -> Vec<Felt> {
        assert_eq!(a.len(), b.len(), "elementwise_gl_mul: length mismatch");
        let n = a.len();
        let bytes_u64 = (n * 2 * core::mem::size_of::<u32>()) as wgpu::BufferAddress;

        // Pack each Felt as [lo, hi] u32 pair (little-endian word order).
        let pack = |xs: &[Felt]| -> Vec<u32> {
            let mut out = Vec::with_capacity(xs.len() * 2);
            for f in xs {
                let v = f.as_canonical_u64();
                out.push(v as u32);
                out.push((v >> 32) as u32);
            }
            out
        };
        let a_packed = pack(a);
        let b_packed = pack(b);

        let kernel = alloc::format!(
            r#"
{prelude}

@group(0) @binding(0) var<storage, read>       a_buf : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       b_buf : array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> c_buf : array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&a_buf)) {{ return; }}
    c_buf[i] = gl_mul(a_buf[i], b_buf[i]);
}}
"#,
            prelude = GOLDILOCKS_WGSL,
        );

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gl_mul kernel"),
            source: wgpu::ShaderSource::Wgsl(kernel.into()),
        });

        let a_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gl_mul.a"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(&a_packed));

        let b_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gl_mul.b"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&b_packed));

        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gl_mul.c"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gl_mul.staging"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("gl_mul pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gl_mul binds"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("gl_mul encoder") },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gl_mul pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes_u64);
        self.queue.submit(Some(encoder.finish()));

        let bytes = read_staging_async(&self.device, &staging).await;
        let raw: &[u32] = bytemuck::cast_slice(&bytes);
        let mut out = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(2) {
            let v = (chunk[0] as u64) | ((chunk[1] as u64) << 32);
            out.push(Felt::new(v));
        }
        out
    }

    /// Forward NTT on each column of a row-major `(rows, cols)` matrix.
    /// `data` is laid out row-major: element at row r col c is `data[r * cols + c]`.
    /// Output is in **bit-reversed row order** (per-column, columns are independent).
    ///
    /// This is the multi-column generalisation of `gl_dft_single_column`.
    /// For now it just loops columns CPU-side; a fused 2D-dispatch kernel
    /// would be faster but is performance-only — correctness is identical.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_dft_batch(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        pollster::block_on(self.gl_dft_batch_async(data, rows, cols))
    }

    /// Async core for multi-column forward NTT.
    ///
    /// One upload of the full row-major matrix, log2(rows) compute dispatches
    /// (each processing all `cols` columns in parallel via 2D dispatch),
    /// one readback. Output is in **bit-reversed row order** (per column);
    /// wrap in `BitReversedMatrixView` for natural-order access.
    pub async fn gl_dft_batch_async(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        assert_eq!(data.len(), rows * cols, "gl_dft_batch: shape mismatch");
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        assert!(rows.is_power_of_two() && rows >= 2, "rows must be power-of-2 >= 2");

        let (data_buf, twid_buf, params_buf, staging) = self.alloc_ntt_buffers(rows, cols);
        let bind_group = self.make_ntt_bind_group(&data_buf, &twid_buf, &params_buf);

        // Upload data + twiddles.
        self.upload_packed_felts(&data_buf, data);
        let twiddles = compute_twiddles(rows);
        self.upload_packed_felts(&twid_buf, &twiddles);

        // Run all log2(rows) DIF stages.
        self.run_ntt_stages(&bind_group, &params_buf, rows, cols);

        // Copy data → staging, single readback.
        let bytes = (rows * cols * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let mut copy_enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("dft_batch readback") },
        );
        copy_enc.copy_buffer_to_buffer(&data_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(copy_enc.finish()));

        decode_packed_felts(read_staging_async(&self.device, &staging).await, rows * cols)
    }

    /// Inverse NTT on each column of a row-major matrix; natural-order in,
    /// natural-order out.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_idft_batch(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        pollster::block_on(self.gl_idft_batch_async(data, rows, cols))
    }

    /// Async core for multi-column inverse NTT.
    ///
    /// Pipeline: forward NTT (in-place on `data_buf`, output in bit-reversed
    /// row order) → `idft_finalize` kernel (writes natural-order `dst[r] =
    /// src_bit_rev[bit_rev((n-r) % n)] / n`) → readback.
    pub async fn gl_idft_batch_async(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        assert_eq!(data.len(), rows * cols, "gl_idft_batch: shape mismatch");
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        assert!(rows.is_power_of_two() && rows >= 2, "rows must be power-of-2 >= 2");

        let (data_buf, twid_buf, params_buf, staging) = self.alloc_ntt_buffers(rows, cols);
        let ntt_bind = self.make_ntt_bind_group(&data_buf, &twid_buf, &params_buf);

        self.upload_packed_felts(&data_buf, data);
        self.upload_packed_felts(&twid_buf, &compute_twiddles(rows));

        self.run_ntt_stages(&ntt_bind, &params_buf, rows, cols);

        // Allocate a destination buffer for the finalized output.
        let bytes = (rows * cols * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idft.dst"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Run the finalize kernel.
        self.run_idft_finalize(&data_buf, &dst_buf, rows, cols);

        // Copy dst → staging, single readback.
        let mut copy_enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("idft_batch readback") },
        );
        copy_enc.copy_buffer_to_buffer(&dst_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(copy_enc.finish()));

        decode_packed_felts(read_staging_async(&self.device, &staging).await, rows * cols)
    }

    /// Coset LDE on each column. Output dims are `(rows << added_bits, cols)`,
    /// row-major, in **bit-reversed row order**.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_coset_lde_batch(
        &self,
        data: &[Felt],
        rows: usize,
        cols: usize,
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        pollster::block_on(self.gl_coset_lde_batch_async(data, rows, cols, added_bits, shift))
    }

    /// Async core for multi-column coset LDE.
    ///
    /// Full pipeline on GPU, single upload + single readback:
    ///   1. forward NTT on input (rows × cols), in place → bit-reversed output
    ///   2. idft_finalize: write natural-order coefficients to a fresh
    ///      lde_buffer (sized rows*2^added_bits × cols), zeroing the upper rows
    ///   3. coset_shift: scale row r by shift^r in the lde_buffer
    ///   4. forward NTT on the lde_buffer (lde_rows × cols), in place
    ///   5. readback the lde_buffer
    pub async fn gl_coset_lde_batch_async(
        &self,
        data: &[Felt],
        rows: usize,
        cols: usize,
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        assert_eq!(data.len(), rows * cols, "gl_coset_lde_batch: shape mismatch");
        let lde_rows = rows << added_bits;
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        assert!(rows.is_power_of_two() && rows >= 2, "rows must be power-of-2 >= 2");
        assert!(lde_rows.is_power_of_two() && lde_rows >= 2);

        // Buffers for the input-sized NTT (iDFT step).
        let small_bytes = (rows * cols * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let lde_bytes = (lde_rows * cols * core::mem::size_of::<u64>()) as wgpu::BufferAddress;

        let small_data = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.small_data"),
            size: small_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let small_twid = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.small_twid"),
            size: ((rows / 2) * core::mem::size_of::<u64>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let small_params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.small_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lde_data = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.lde_data"),
            size: lde_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lde_twid = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.lde_twid"),
            size: ((lde_rows / 2) * core::mem::size_of::<u64>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lde_params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.lde_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.staging"),
            size: lde_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload input + small/lde twiddles.
        self.upload_packed_felts(&small_data, data);
        self.upload_packed_felts(&small_twid, &compute_twiddles(rows));
        self.upload_packed_felts(&lde_twid, &compute_twiddles(lde_rows));

        // Step 1: forward NTT on small_data.
        let small_bind = self.make_ntt_bind_group(&small_data, &small_twid, &small_params);
        self.run_ntt_stages(&small_bind, &small_params, rows, cols);

        // Step 2: idft_finalize, writing into the lde_data buffer's first
        // rows*cols Felts, with the rest zero-cleared via clear_buffer.
        // The kernel only writes rows*cols entries; the rest remains zero.
        {
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("coset_lde.zero+finalize") },
            );
            // Zero the entire LDE buffer first (fast device-side clear).
            enc.clear_buffer(&lde_data, 0, None);
            self.queue.submit(Some(enc.finish()));
        }
        // Run finalize: src=small_data (bit-reversed DFT), dst=lde_data
        // (natural-order coefficients in the first `rows` rows).
        self.run_idft_finalize(&small_data, &lde_data, rows, cols);

        // Step 3: coset shift on the lde_data buffer.
        // shift_powers = [1, shift, shift^2, ..., shift^{lde_rows-1}]
        let mut shift_powers: Vec<Felt> = Vec::with_capacity(lde_rows);
        let mut acc = Felt::ONE;
        for _ in 0..lde_rows {
            shift_powers.push(acc);
            acc *= shift;
        }
        let shift_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_lde.shift_powers"),
            size: (lde_rows * core::mem::size_of::<u64>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.upload_packed_felts(&shift_buf, &shift_powers);
        self.run_coset_shift(&lde_data, &shift_buf, lde_rows, cols);

        // Step 4: forward NTT on lde_data (now coset-shifted coefficients).
        let lde_bind = self.make_ntt_bind_group(&lde_data, &lde_twid, &lde_params);
        self.run_ntt_stages(&lde_bind, &lde_params, lde_rows, cols);

        // Step 5: readback.
        let mut copy_enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("coset_lde readback") },
        );
        copy_enc.copy_buffer_to_buffer(&lde_data, 0, &staging, 0, lde_bytes);
        self.queue.submit(Some(copy_enc.finish()));

        decode_packed_felts(read_staging_async(&self.device, &staging).await, lde_rows * cols)
    }

    /// Native sync wrapper.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_dft_single_column(&self, input: &[Felt]) -> Vec<Felt> {
        pollster::block_on(self.gl_dft_single_column_async(input))
    }

    /// Forward NTT on a single column. Now a thin wrapper around the multi-
    /// column `gl_dft_batch_async` with `cols=1`. Output is in **bit-reversed
    /// order** — wrap in `BitReversedMatrixView` for natural-order access.
    pub async fn gl_dft_single_column_async(&self, input: &[Felt]) -> Vec<Felt> {
        self.gl_dft_batch_async(input, input.len(), 1).await
    }

    /// Native sync wrapper.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_idft_single_column(&self, input: &[Felt]) -> Vec<Felt> {
        pollster::block_on(self.gl_idft_single_column_async(input))
    }

    /// Inverse NTT on a single column. Now a thin wrapper around the multi-
    /// column `gl_idft_batch_async` with `cols=1`.
    pub async fn gl_idft_single_column_async(&self, input: &[Felt]) -> Vec<Felt> {
        self.gl_idft_batch_async(input, input.len(), 1).await
    }

    /// Native sync wrapper.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_coset_lde_single_column(
        &self,
        input: &[Felt],
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        pollster::block_on(self.gl_coset_lde_single_column_async(input, added_bits, shift))
    }

    /// Coset LDE on a single column. Thin wrapper around the multi-column
    /// `gl_coset_lde_batch_async` with `cols=1`.
    pub async fn gl_coset_lde_single_column_async(
        &self,
        input: &[Felt],
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        self.gl_coset_lde_batch_async(input, input.len(), 1, added_bits, shift).await
    }

    // ---- multi-column kernel helpers ----------------------------------------

    /// Allocate the `(data, twiddles, params, staging)` quartet for an NTT
    /// over `(rows, cols)`. Buffers are sized but not populated.
    fn alloc_ntt_buffers(
        &self,
        rows: usize,
        cols: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let data_bytes = (rows * cols * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let twid_bytes = ((rows / 2) * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let data = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.data"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let twid = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.twiddles"),
            size: twid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (data, twid, params, staging)
    }

    /// Pack `Felt` slice as `(lo, hi)` u32 pairs and `queue.write_buffer` it
    /// into `buf` at offset 0.
    fn upload_packed_felts(&self, buf: &wgpu::Buffer, xs: &[Felt]) {
        let mut packed: Vec<u32> = Vec::with_capacity(xs.len() * 2);
        for f in xs {
            let v = f.as_canonical_u64();
            packed.push(v as u32);
            packed.push((v >> 32) as u32);
        }
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Build the NTT bind group from a (data, twiddles, params) triple.
    fn make_ntt_bind_group(
        &self,
        data: &wgpu::Buffer,
        twiddles: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ntt binds"),
            layout: &self.ntt_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: twiddles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        })
    }

    /// Run all log2(rows) DIF butterfly stages on the matrix bound by
    /// `bind_group`. Each stage updates `params` then dispatches a
    /// `(n_half / 64, cols, 1)` workgroup grid.
    fn run_ntt_stages(
        &self,
        bind_group: &wgpu::BindGroup,
        params_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let log_n = rows.trailing_zeros();
        let n_half = (rows / 2) as u32;
        let cols_u32 = cols as u32;
        let dispatch_x = (n_half as usize).div_ceil(64) as u32;
        let dispatch_y = cols_u32;

        // Submit ALL stages in one command buffer to avoid per-stage overhead.
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("ntt_stages") },
        );
        for stage in 0..log_n {
            let half = (rows >> (stage + 1)) as u32;
            let params: [u32; 4] = [half, stage, n_half, cols_u32];
            // queue.write_buffer is on the queue; for proper ordering we need
            // to flush the current encoder before the writes. To keep stages in
            // one encoder, write params synchronously between submits — i.e.
            // submit the previous batch, write, encode the next.
            //
            // Workaround: split into one submit per stage for now (still
            // batched within a single queue.submit per stage call). Acceptable
            // — submit overhead is small compared to per-stage dispatch.
            // (A fully fused multi-stage encoder is possible with multiple
            // uniform buffers, but for correctness-equivalence let's keep this.)
            let _ = encoder; // pacify unused-var if we abort here
            self.queue.write_buffer(params_buf, 0, bytemuck::cast_slice(&params));
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ntt stage") },
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ntt stage pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ntt_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            self.queue.submit(Some(enc.finish()));
            // re-bind encoder var so the unused warning above doesn't fire on
            // the last iteration.
            encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ntt_stages_tail") },
            );
        }
        let _ = encoder;
    }

    /// Run the iDFT finalize kernel: src (bit-reversed DFT) → dst (natural-
    /// order iDFT, scaled by n^{-1}, with the j → (n-j)%n permutation).
    fn run_idft_finalize(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let log_n = rows.trailing_zeros();
        let n_inv = Felt::new(rows as u64)
            .try_inverse()
            .expect("Goldilocks n must be invertible")
            .as_canonical_u64();

        let params: [u32; 8] = [
            rows as u32,
            log_n,
            cols as u32,
            n_inv as u32,
            (n_inv >> 32) as u32,
            0,
            0,
            0,
        ];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idft_finalize.params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("idft_finalize binds"),
            layout: &self.idft_finalize_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("idft_finalize") },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idft_finalize pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.idft_finalize_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let dispatch_x = rows.div_ceil(64) as u32;
            pass.dispatch_workgroups(dispatch_x, cols as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Run the coset_shift kernel: `data[r, c] *= shift_powers[r]` in place.
    fn run_coset_shift(
        &self,
        data: &wgpu::Buffer,
        shift_powers: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, 0, 0];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coset_shift.params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("coset_shift binds"),
            layout: &self.coset_shift_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shift_powers.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("coset_shift") },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("coset_shift pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.coset_shift_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let dispatch_x = rows.div_ceil(64) as u32;
            pass.dispatch_workgroups(dispatch_x, cols as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn identity_setup(
        &self,
        input: &[u32],
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::ComputePipeline,
        wgpu::BindGroup,
        wgpu::BufferAddress,
    ) {
        let bytes = input.len() * core::mem::size_of::<u32>();
        let bytes_u64 = bytes as wgpu::BufferAddress;

        const SHADER: &str = r#"
            @group(0) @binding(0) var<storage, read>       input  : array<u32>;
            @group(0) @binding(1) var<storage, read_write> output : array<u32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= arrayLength(&input)) { return; }
                output[i] = input[i];
            }
        "#;

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("identity copy"),
            source: wgpu::ShaderSource::Wgsl(SHADER.to_owned().into()),
        });

        let input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(input));

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("identity pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("identity binds"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
            ],
        });

        (input_buf, output_buf, staging, pipeline, bind_group, bytes_u64)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn buffer_roundtrip_identity() {
        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };
        let input: Vec<u32> = (0..1024).collect();
        let output = ctx.buffer_roundtrip_u32(&input);
        assert_eq!(input, output, "identity kernel produced different bytes");
    }

    #[test]
    fn gl_dft_single_column_matches_cpu() {
        use miden_crypto::stark::dft::{Radix2DitParallel, TwoAdicSubgroupDft};
        use p3_matrix::Matrix;
        use p3_matrix::dense::RowMajorMatrix;
        use rand::Rng;

        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        // Small N for first correctness pass.
        for log_n in [3, 6, 10] {
            let n: usize = 1 << log_n;
            let input: Vec<Felt> = (0..n).map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER)).collect();

            // GPU: produces bit-reversed-order output.
            let gpu_bit_rev = ctx.gl_dft_single_column(&input);

            // CPU ground truth via Radix2DitParallel.
            // dft_batch returns BitReversedMatrixView<RowMajorMatrix<Felt>>.
            // .to_row_major_matrix() un-permutes to natural order.
            let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
            let mat = RowMajorMatrix::new_col(input.clone());
            let cpu_natural = cpu.dft_batch(mat).to_row_major_matrix().values;

            // Convert GPU bit-reversed buffer to natural order so we can compare.
            let mut gpu_natural = vec![Felt::ZERO; n];
            let log_n_usize = log_n as usize;
            for r in 0..n {
                let r_rev = reverse_bits_len(r, log_n_usize);
                gpu_natural[r] = gpu_bit_rev[r_rev];
            }

            for i in 0..n {
                assert_eq!(
                    gpu_natural[i].as_canonical_u64(),
                    cpu_natural[i].as_canonical_u64(),
                    "NTT mismatch at log_n={log_n} i={i}: gpu={} cpu={}",
                    gpu_natural[i].as_canonical_u64(),
                    cpu_natural[i].as_canonical_u64(),
                );
            }
        }
    }

    #[test]
    fn gl_idft_single_column_matches_cpu() {
        use miden_crypto::stark::dft::{Radix2DitParallel, TwoAdicSubgroupDft};
        use p3_matrix::dense::RowMajorMatrix;
        use rand::Rng;

        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        for log_n in [3, 6, 10] {
            let n: usize = 1 << log_n;
            let input: Vec<Felt> = (0..n).map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER)).collect();

            let gpu = ctx.gl_idft_single_column(&input);

            let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
            let mat = RowMajorMatrix::new_col(input.clone());
            let cpu_idft = cpu.idft_batch(mat).values;

            for i in 0..n {
                assert_eq!(
                    gpu[i].as_canonical_u64(),
                    cpu_idft[i].as_canonical_u64(),
                    "iNTT mismatch at log_n={log_n} i={i}: gpu={} cpu={}",
                    gpu[i].as_canonical_u64(),
                    cpu_idft[i].as_canonical_u64(),
                );
            }
        }
    }

    /// Multi-column shape: ensures gl_dft_batch / gl_idft_batch / gl_coset_lde_batch
    /// produce the exact bytes Radix2DitParallel does on row-major matrices with
    /// width > 1. Single-column tests already cover the math; this gates the
    /// shape conversion (column extract / re-pack).
    #[test]
    fn gl_dft_batch_matches_cpu_multicol() {
        use miden_crypto::stark::dft::{Radix2DitParallel, TwoAdicSubgroupDft};
        use p3_matrix::Matrix;
        use p3_matrix::dense::RowMajorMatrix;
        use rand::Rng;

        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        let rows = 64;
        let cols = 5;
        let data: Vec<Felt> = (0..rows * cols).map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER)).collect();

        // dft_batch
        let gpu_dft_bit_rev = ctx.gl_dft_batch(&data, rows, cols);
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
        let cpu_dft_natural = cpu
            .dft_batch(RowMajorMatrix::new(data.clone(), cols))
            .to_row_major_matrix()
            .values;
        let log_rows = rows.trailing_zeros() as usize;
        for r in 0..rows {
            let r_rev = reverse_bits_len(r, log_rows);
            for c in 0..cols {
                let gpu_v = gpu_dft_bit_rev[r_rev * cols + c].as_canonical_u64();
                let cpu_v = cpu_dft_natural[r * cols + c].as_canonical_u64();
                assert_eq!(gpu_v, cpu_v, "dft_batch mismatch r={r} c={c}");
            }
        }

        // idft_batch
        let gpu_idft = ctx.gl_idft_batch(&data, rows, cols);
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
        let cpu_idft = cpu.idft_batch(RowMajorMatrix::new(data.clone(), cols)).values;
        for i in 0..rows * cols {
            assert_eq!(gpu_idft[i].as_canonical_u64(), cpu_idft[i].as_canonical_u64(), "idft_batch mismatch i={i}");
        }

        // coset_lde_batch
        let added_bits = 2;
        let shift = Felt::new(7);
        let gpu_lde_bit_rev = ctx.gl_coset_lde_batch(&data, rows, cols, added_bits, shift);
        let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
        let cpu_lde_natural = cpu
            .coset_lde_batch(RowMajorMatrix::new(data.clone(), cols), added_bits, shift)
            .to_row_major_matrix()
            .values;
        let lde_rows = rows << added_bits;
        let log_lde = lde_rows.trailing_zeros() as usize;
        for r in 0..lde_rows {
            let r_rev = reverse_bits_len(r, log_lde);
            for c in 0..cols {
                let gpu_v = gpu_lde_bit_rev[r_rev * cols + c].as_canonical_u64();
                let cpu_v = cpu_lde_natural[r * cols + c].as_canonical_u64();
                assert_eq!(gpu_v, cpu_v, "coset_lde_batch mismatch r={r} c={c}");
            }
        }
    }

    #[test]
    fn gl_coset_lde_single_column_matches_cpu() {
        use miden_crypto::stark::dft::{Radix2DitParallel, TwoAdicSubgroupDft};
        use p3_matrix::Matrix;
        use p3_matrix::dense::RowMajorMatrix;
        use rand::Rng;

        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        // Smaller sizes — coset_lde inflates by 2^added_bits.
        let cases: &[(usize, usize)] = &[(3, 3), (6, 3), (10, 2)];
        for &(log_n, added_bits) in cases {
            let n: usize = 1 << log_n;
            let input: Vec<Felt> = (0..n).map(|_| Felt::new(rng.random::<u64>() % Felt::ORDER)).collect();
            let shift = Felt::new(7);

            // GPU produces bit-reversed-order LDE output.
            let gpu_bit_rev = ctx.gl_coset_lde_single_column(&input, added_bits, shift);

            // CPU ground truth — natural-order via .to_row_major_matrix() on
            // the BitReversedMatrixView.
            let cpu: Radix2DitParallel<Felt> = Radix2DitParallel::default();
            let mat = RowMajorMatrix::new_col(input.clone());
            let cpu_natural = cpu.coset_lde_batch(mat, added_bits, shift).to_row_major_matrix().values;

            // Permute GPU output to natural order.
            let lde_n = n << added_bits;
            let log_lde_n = lde_n.trailing_zeros() as usize;
            let mut gpu_natural = vec![Felt::ZERO; lde_n];
            for r in 0..lde_n {
                gpu_natural[r] = gpu_bit_rev[reverse_bits_len(r, log_lde_n)];
            }

            for i in 0..lde_n {
                assert_eq!(
                    gpu_natural[i].as_canonical_u64(),
                    cpu_natural[i].as_canonical_u64(),
                    "coset_lde mismatch at log_n={log_n} added={added_bits} i={i}: gpu={} cpu={}",
                    gpu_natural[i].as_canonical_u64(),
                    cpu_natural[i].as_canonical_u64(),
                );
            }
        }
    }


    #[test]
    fn elementwise_gl_mul_matches_cpu() {
        use rand::Rng;
        let ctx = match pollster::block_on(WgpuContext::new()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no wgpu adapter available ({e})");
                return;
            }
        };

        let mut rng = rand::rng();
        let n: usize = 1024;
        // Mix of: zero, one, p-1, random, and a few hand-picked values
        // to exercise reduction edge cases.
        let mut a: Vec<Felt> = Vec::with_capacity(n);
        let mut b: Vec<Felt> = Vec::with_capacity(n);
        a.push(Felt::new(0));                  b.push(Felt::new(0));
        a.push(Felt::new(1));                  b.push(Felt::new(0xFFFFFFFF_00000000));
        a.push(Felt::new(0xFFFFFFFF_00000000));b.push(Felt::new(0xFFFFFFFF_00000000));
        a.push(Felt::new(0xFFFFFFFE_00000001));b.push(Felt::new(0xFFFFFFFE_00000001));
        a.push(Felt::new(2));                  b.push(Felt::new(2));
        for _ in a.len()..n {
            a.push(Felt::new(rng.random::<u64>() % Felt::ORDER));
            b.push(Felt::new(rng.random::<u64>() % Felt::ORDER));
        }

        let gpu = ctx.elementwise_gl_mul(&a, &b);
        let cpu: Vec<Felt> = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect();

        for i in 0..n {
            assert_eq!(
                gpu[i].as_canonical_u64(),
                cpu[i].as_canonical_u64(),
                "gl_mul mismatch at i={i}: a={} b={} gpu={} cpu={}",
                a[i].as_canonical_u64(),
                b[i].as_canonical_u64(),
                gpu[i].as_canonical_u64(),
                cpu[i].as_canonical_u64(),
            );
        }
    }
}
