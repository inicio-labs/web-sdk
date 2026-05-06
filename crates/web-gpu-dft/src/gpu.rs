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

use crate::GpuInitError;

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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn buffer_roundtrip_u32(&self, input: &[u32]) -> Vec<u32> {
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
        // Keep the optimizer from dropping these before the queue runs.
        drop(input_buf);

        // Synchronous readback via callback + device.poll(Wait). Mutex (not
        // Cell) because wgpu's map_async callback is Send-bound on native.
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
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
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
}
