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
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

use miden_crypto::Felt;
use p3_field::{Field, TwoAdicField};

use crate::GpuInitError;

/// Goldilocks WGSL primitives (mul_u32, mul_u64_to_u128, gl_reduce_u128, gl_add, gl_mul).
/// Concatenated into the front of every kernel that does Goldilocks math.
const GOLDILOCKS_WGSL: &str = include_str!("shaders/goldilocks.wgsl");

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

/// Owns a wgpu device + queue. Cheap to clone if wrapped in `Arc`.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("miden-web-gpu-dft device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| GpuInitError::DeviceInit(e.to_string()))?;

        Ok(Self { device, queue })
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

        let mapped: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let mapped_cb = mapped.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            *mapped_cb.lock().unwrap() = Some(res);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device poll");
        mapped
            .lock()
            .unwrap()
            .take()
            .expect("map_async callback did not fire")
            .expect("map_async failed");

        let data = staging.slice(..).get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(2) {
            let v = (chunk[0] as u64) | ((chunk[1] as u64) << 32);
            out.push(Felt::new(v));
        }
        drop(data);
        staging.unmap();
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
        assert_eq!(data.len(), rows * cols, "gl_dft_batch: shape mismatch");
        if cols == 0 {
            return Vec::new();
        }
        // Extract each column, run the single-column kernel, write back.
        let mut out = vec![Felt::ZERO; rows * cols];
        let mut col_buf = vec![Felt::ZERO; rows];
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = data[r * cols + c];
            }
            let dft = self.gl_dft_single_column(&col_buf);
            for r in 0..rows {
                out[r * cols + c] = dft[r];
            }
        }
        out
    }

    /// Inverse NTT on each column of a row-major matrix; natural-order in,
    /// natural-order out.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_idft_batch(&self, data: &[Felt], rows: usize, cols: usize) -> Vec<Felt> {
        assert_eq!(data.len(), rows * cols, "gl_idft_batch: shape mismatch");
        if cols == 0 {
            return Vec::new();
        }
        let mut out = vec![Felt::ZERO; rows * cols];
        let mut col_buf = vec![Felt::ZERO; rows];
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = data[r * cols + c];
            }
            let idft = self.gl_idft_single_column(&col_buf);
            for r in 0..rows {
                out[r * cols + c] = idft[r];
            }
        }
        out
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
        assert_eq!(data.len(), rows * cols, "gl_coset_lde_batch: shape mismatch");
        let lde_rows = rows << added_bits;
        if cols == 0 {
            return Vec::new();
        }
        let mut out = vec![Felt::ZERO; lde_rows * cols];
        let mut col_buf = vec![Felt::ZERO; rows];
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = data[r * cols + c];
            }
            let lde = self.gl_coset_lde_single_column(&col_buf, added_bits, shift);
            for r in 0..lde_rows {
                out[r * cols + c] = lde[r];
            }
        }
        out
    }

    /// Forward NTT (Cooley-Tukey decimation-in-frequency, radix-2) on a single
    /// column. Output is in **bit-reversed order** — wrap in a
    /// `BitReversedMatrixView` for natural-order access.
    ///
    /// Native-only; wasm32 sibling lands later. Used as the correctness gate
    /// for the WGSL NTT kernel, diffed against `Radix2DitParallel.dft_batch`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_dft_single_column(&self, input: &[Felt]) -> Vec<Felt> {
        let n = input.len();
        assert!(n.is_power_of_two() && n >= 2, "NTT length must be power-of-2 >= 2");
        let log_n = n.trailing_zeros();

        // Precompute powers of the primitive n-th root of unity:
        // twiddles[k] = w_n^k for k in [0, n/2).
        let w_n = Felt::two_adic_generator(log_n as usize);
        let mut twiddles: Vec<Felt> = Vec::with_capacity(n / 2);
        let mut acc = Felt::ONE;
        for _ in 0..n / 2 {
            twiddles.push(acc);
            acc *= w_n;
        }

        // Pack to (lo, hi) u32 pairs for the WGSL kernel.
        let pack = |xs: &[Felt]| -> Vec<u32> {
            let mut out = Vec::with_capacity(xs.len() * 2);
            for f in xs {
                let v = f.as_canonical_u64();
                out.push(v as u32);
                out.push((v >> 32) as u32);
            }
            out
        };
        let data_packed = pack(input);
        let twiddles_packed = pack(&twiddles);

        let data_bytes = (n * 2 * core::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let twid_bytes = (twiddles.len() * 2 * core::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let kernel = alloc::format!(
            r#"
{prelude}

struct NttParams {{
    half     : u32,
    log_step : u32,
    n_half   : u32,
    pad      : u32,
}};

@group(0) @binding(0) var<storage, read_write> data     : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       twiddles : array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params   : NttParams;

@compute @workgroup_size(64)
fn ntt_stage(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let butterfly = gid.x;
    if (butterfly >= params.n_half) {{ return; }}

    let half = params.half;
    let group_size = half * 2u;
    let group = butterfly / half;
    let j     = butterfly % half;
    let i_a   = group * group_size + j;
    let i_b   = i_a + half;

    let a_val = data[i_a];
    let b_val = data[i_b];

    // DIF butterfly: a' = a + b, b' = (a - b) * twiddle[j * stride]
    let stride = 1u << params.log_step;
    let tw_idx = j * stride;
    let tw     = twiddles[tw_idx];

    let sum   = gl_add(a_val, b_val);
    let diff  = gl_sub(a_val, b_val);
    let mul_v = gl_mul(diff, tw);

    data[i_a] = sum;
    data[i_b] = mul_v;
}}
"#,
            prelude = GOLDILOCKS_WGSL,
        );

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ntt_stage kernel"),
            source: wgpu::ShaderSource::Wgsl(kernel.into()),
        });

        let data_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.data"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&data_buf, 0, bytemuck::cast_slice(&data_packed));

        let twid_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.twiddles"),
            size: twid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&twid_buf, 0, bytemuck::cast_slice(&twiddles_packed));

        // 16-byte aligned uniform buffer holding NttParams.
        let params_size = 16u64;
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.params"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt.staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("ntt pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("ntt_stage"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ntt binds"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: twid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        // Run log_n stages of butterflies.
        let n_half = (n / 2) as u32;
        for stage in 0..log_n {
            let half = (n >> (stage + 1)) as u32;
            // Pack params as 4 u32s = 16 bytes.
            let params: [u32; 4] = [half, stage as u32, n_half, 0];
            self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ntt stage encoder") },
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ntt stage pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n_half as usize).div_ceil(64) as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            // No need to poll between stages — ordering on a single queue is
            // preserved; the next dispatch sees the previous one's writes.
        }

        // Final readback.
        let mut copy_enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("ntt copy encoder") },
        );
        copy_enc.copy_buffer_to_buffer(&data_buf, 0, &staging, 0, data_bytes);
        self.queue.submit(Some(copy_enc.finish()));

        let mapped: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let mapped_cb = mapped.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            *mapped_cb.lock().unwrap() = Some(res);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device poll");
        mapped
            .lock()
            .unwrap()
            .take()
            .expect("map_async callback did not fire")
            .expect("map_async failed");

        let data = staging.slice(..).get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(2) {
            let v = (chunk[0] as u64) | ((chunk[1] as u64) << 32);
            out.push(Felt::new(v));
        }
        drop(data);
        staging.unmap();
        out
    }

    /// Inverse NTT on a single column. Natural-order input → natural-order
    /// output. Implementation follows Plonky3's `idft_batch` default impl:
    /// run the forward NTT, divide by n, swap rows 1..n/2 with rows n-1..n/2.
    /// (Mathematically: iDFT(X)[j] = X[(-j) mod n] / n in cyclic-group NTT.)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_idft_single_column(&self, input: &[Felt]) -> Vec<Felt> {
        let n = input.len();
        assert!(n.is_power_of_two() && n >= 2, "iNTT length must be power-of-2 >= 2");
        let log_n = n.trailing_zeros() as usize;

        // Forward NTT (bit-reversed-order output).
        let bit_rev = self.gl_dft_single_column(input);

        // 1. bit-reverse permute → natural-order DFT.
        let mut natural = alloc::vec![Felt::ZERO; n];
        for i in 0..n {
            natural[i] = bit_rev[reverse_bits_len(i, log_n)];
        }

        // 2. divide by n (multiply by n^{-1} in the field).
        let n_inv = Felt::new(n as u64)
            .try_inverse()
            .expect("Goldilocks n must be invertible (p coprime to n)");
        for x in natural.iter_mut() {
            *x = *x * n_inv;
        }

        // 3. Swap rows 1..n/2 with rows n-1..n/2. Equivalent to mapping
        // index j → (n - j) mod n (with row 0 fixed) — the iDFT permutation.
        for r in 1..n / 2 {
            natural.swap(r, n - r);
        }

        natural
    }

    /// Coset LDE on a single column: extends `n`-element coefficient vector
    /// (interpreted as evaluations on the standard subgroup of order n) onto
    /// the coset `shift * K` where K is the order-`n << added_bits` subgroup.
    ///
    /// Output is in **bit-reversed order** matching `BitReversedMatrixView`
    /// storage convention. Pipeline: iNTT (natural coefficients) → resize
    /// with zeros to n*2^added_bits → coset shift on CPU (multiply coef i
    /// by shift^i) → forward NTT.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gl_coset_lde_single_column(
        &self,
        input: &[Felt],
        added_bits: usize,
        shift: Felt,
    ) -> Vec<Felt> {
        let n = input.len();
        assert!(n.is_power_of_two() && n >= 2, "coset_lde input length must be power-of-2 >= 2");

        // 1. iDFT (natural → coefficients).
        let mut coefs = self.gl_idft_single_column(input);

        // 2. Resize with zeros to n * 2^added_bits.
        let lde_n = n << added_bits;
        coefs.resize(lde_n, Felt::ZERO);

        // 3. Coset shift: scale coef[i] by shift^i. Equivalent to evaluating
        // the polynomial on `shift * K` instead of K.
        let mut shift_pow = Felt::ONE;
        for c in coefs.iter_mut() {
            *c = *c * shift_pow;
            shift_pow *= shift;
        }

        // 4. Forward NTT on the extended buffer (bit-reversed output).
        self.gl_dft_single_column(&coefs)
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
