//! Minimal wgpu plumbing for Phase 0b GPU parity tests.
//!
//! Phase 0b Unit 6b: dispatch a kernel that runs `qf_mul` on input pairs
//! and compare the GPU output to the canonical `BinomialExtensionField`
//! result on the host.
//!
//! Phase 0b Unit 7+ will grow this module into the full tape-interpreter
//! kernel + buffer pipeline. For now the surface is intentionally small.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::encode::{limbs_to_quadfelt, quadfelt_to_limbs};
use crate::recorder::QuadFelt;

const GOLDILOCKS_WGSL: &str =
    include_str!("../../web-gpu-dft/src/shaders/goldilocks.wgsl");
const QUADFELT_WGSL: &str = include_str!("shaders/quadfelt.wgsl");

/// One (a, b) pair of QuadFelts, packed for upload as a `vec4<u32>` (a) +
/// `vec4<u32>` (b) = 8 u32s. Each input slot in the kernel reads two
/// `vec4<u32>`s.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QfPair {
    a: [u32; 4],
    b: [u32; 4],
}

/// Dispatch a wgpu kernel that runs `qf_mul(a, b)` for each input pair and
/// returns the resulting QuadFelts.
///
/// Native-only for Phase 0b (run via `cargo test --features real-gpu` on
/// macOS = Metal, Linux = Vulkan/llvmpipe). The web build will use the
/// existing SAB bridge in web-gpu-dft to dispatch from the WASM worker; that
/// integration lands in Phase 3.
pub async fn qf_mul_pairs(pairs: &[(QuadFelt, QuadFelt)]) -> Vec<QuadFelt> {
    let n = pairs.len() as u32;
    assert!(n > 0, "qf_mul_pairs: empty input");

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter request failed");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("miden-web-gpu-air qf_mul test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::default(),
        })
        .await
        .expect("wgpu device request failed");
    let device = Arc::new(device);

    // Pack inputs.
    let inputs: Vec<QfPair> = pairs
        .iter()
        .map(|(a, b)| QfPair {
            a: quadfelt_to_limbs(*a),
            b: quadfelt_to_limbs(*b),
        })
        .collect();
    let in_bytes: &[u8] = bytemuck::cast_slice(&inputs);
    let out_byte_len = (n as u64) * 16; // each output = vec4<u32> = 16 bytes

    let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qf_mul.in"),
        size: in_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&in_buf, 0, in_bytes);

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qf_mul.out"),
        size: out_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qf_mul.staging"),
        size: out_byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Build kernel by concatenating: goldilocks.wgsl + quadfelt.wgsl + kernel.
    let kernel_src = format!(
        "{goldilocks}\n\n{quadfelt}\n\n\
struct QfPair {{ a: vec4<u32>, b: vec4<u32> }};\n\
@group(0) @binding(0) var<storage, read>       in_pairs: array<QfPair>;\n\
@group(0) @binding(1) var<storage, read_write> out:      array<vec4<u32>>;\n\
\n\
@compute @workgroup_size(64)\n\
fn qf_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&out)) {{ return; }}\n\
    let p = in_pairs[i];\n\
    out[i] = qf_mul(p.a, p.b);\n\
}}\n",
        goldilocks = GOLDILOCKS_WGSL,
        quadfelt = QUADFELT_WGSL,
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("qf_mul_kernel"),
        source: wgpu::ShaderSource::Wgsl(kernel_src.into()),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("qf_mul.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("qf_mul.pl"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("qf_mul.pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("qf_mul_kernel"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qf_mul.bg"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("qf_mul.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qf_mul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(64);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_byte_len);
    queue.submit([encoder.finish()]);

    // Read back via mapAsync.
    let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device poll failed");
    rx.await.expect("map_async cancelled").expect("map_async failed");

    let bytes = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();

    let out_limbs: &[u32] = bytemuck::cast_slice(&bytes);
    debug_assert_eq!(out_limbs.len(), (n as usize) * 4);

    out_limbs
        .chunks_exact(4)
        .map(|c| limbs_to_quadfelt([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::encode::testing::random_quadfelt;

    /// Phase 0b Unit 6b: byte-for-byte parity between WGSL `qf_mul` and the
    /// canonical p3-field `BinomialExtensionField` mul. 1024 random pairs.
    #[test]
    fn qf_mul_matches_canonical() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let pairs: Vec<(QuadFelt, QuadFelt)> = (0..1024)
            .map(|_| (random_quadfelt(&mut rng), random_quadfelt(&mut rng)))
            .collect();
        let expected: Vec<QuadFelt> = pairs.iter().map(|(a, b)| *a * *b).collect();
        let actual = pollster::block_on(qf_mul_pairs(&pairs));
        assert_eq!(actual.len(), expected.len());
        for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "mismatch at pair {i}: a={:?} b={:?} got={got:?} want={want:?}", pairs[i].0, pairs[i].1);
        }
    }
}
