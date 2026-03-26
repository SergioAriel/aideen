//! gpu_backend — WgpuBlockBackend and CpuBlockBackend
//!
//! - `CpuBlockBackend`: pure nalgebra implementation (always compiled).
//!   Works without GPU, useful for tests and fallback.
//!
//! - `WgpuBlockBackend`: wgpu+WGSL implementation (feature "wgpu").
//!   Uses the `mamba.wgsl` shader from `aideen-block` to run the SSM
//!   on GPU (Metal/Vulkan/WebGPU). Synchronization via pollster::block_on.

use aideen_core::block_backend::BlockBackend;

// ── CPU Fallback ──────────────────────────────────────────────────────────────

/// Pure CPU backend. Implements the same ZOH as MambaSlotReasoning
/// but as a swappable backend. Useful for tests and deployment without GPU.
pub struct CpuBlockBackend;

impl BlockBackend for CpuBlockBackend {
    fn mamba_batch_step(
        &mut self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
    ) -> Result<Vec<f32>, String> {
        let d = x.len();
        if a.len() != d || b.len() != d || c.len() != d || dt.len() != d {
            return Err(format!(
                "CpuBlockBackend: dim mismatch: x={d}, a={}, b={}, c={}, dt={}",
                a.len(),
                b.len(),
                c.len(),
                dt.len()
            ));
        }

        // ZOH discretization: A_bar = exp(dt * A), B_bar = (A_bar - 1) / A * B
        // h_new = A_bar * h_prev + B_bar * x    (h_prev = 0 para un solo paso)
        // y = C * h_new
        let y: Vec<f32> = (0..d)
            .map(|i| {
                let a_bar = (dt[i] * a[i]).exp();
                let b_bar = if a[i].abs() > 1e-8 {
                    (a_bar - 1.0) / a[i] * b[i]
                } else {
                    dt[i] * b[i] // límite cuando A → 0
                };
                let h_new = b_bar * x[i]; // h_prev=0 para paso único
                c[i] * h_new
            })
            .collect();

        Ok(y)
    }
}

// ── GPU Backend (feature = "wgpu") ────────────────────────────────────────────

/// wgpu backend: executes the `mamba_parallel_scan` kernel from aideen-block
/// on GPU via blocking synchronization with `pollster`.
///
/// ## Usage
/// ```ignore
/// let gpu = WgpuBlockBackend::new_blocking();
/// if let Some(mut backend) = gpu {
///     let y = backend.mamba_batch_step(&x, &dt, &a, &b, &c)?;
/// }
/// ```
#[cfg(feature = "wgpu")]
pub struct WgpuBlockBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bridge: aideen_block::mamba::RustMambaBridge,
}

#[cfg(feature = "wgpu")]
impl WgpuBlockBackend {
    /// Initializes the GPU backend synchronously using pollster.
    /// Returns `None` if no GPU is available.
    pub fn new_blocking() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        use aideen_block::{mamba::RustMambaBridge, ComputeState};
        let state = ComputeState::new().await?;
        let bridge = RustMambaBridge::new(&state.device);
        Some(Self {
            device: state.device,
            queue: state.queue,
            bridge,
        })
    }
}

#[cfg(feature = "wgpu")]
impl BlockBackend for WgpuBlockBackend {
    fn mamba_batch_step(
        &mut self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
    ) -> Result<Vec<f32>, String> {
        use bytemuck;
        use wgpu::util::DeviceExt;

        let d = x.len() as u32;
        if d == 0 {
            return Err("WgpuBlockBackend: d_model=0".to_string());
        }

        // Shape uniform: batch=1, seq_len=1, d_model=d, d_state=1
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ComputeShape {
            batch: u32,
            seq: u32,
            d_model: u32,
            d_state: u32,
        }

        let shape = ComputeShape {
            batch: 1,
            seq: 1,
            d_model: d,
            d_state: 1,
        };

        let make_uniform = |data: &[u8]| -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };

        let make_storage_ro = |data: &[f32]| -> wgpu::Buffer {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        // Output buffer: read_write + MAP_READ para poder leer el resultado
        let out_size = (d as usize * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Crear buffers de entrada
        let shape_buf = make_uniform(bytemuck::bytes_of(&shape));
        let x_buf = make_storage_ro(x);
        let dt_buf = make_storage_ro(dt);
        let a_buf = make_storage_ro(a);
        let b_buf = make_storage_ro(b);
        let c_buf = make_storage_ro(c);

        // Bind group acorde al layout de RustMambaBridge
        // (shape, X_in, dt, A, B, C, Y_out)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bridge.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        // Encode + dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bridge.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // workgroup_size(256,1,1), 1 token → 1 workgroup en X, d_model workgroups en Y
            pass.dispatch_workgroups(1, d, 1);
        }

        // Copiar resultado a readback buffer
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Readback síncrono usando pollster
        let result = pollster::block_on(async {
            let (sender, receiver) = futures::channel::oneshot::channel();
            readback_buf
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = sender.send(r);
                });
            self.device.poll(wgpu::Maintain::Wait);
            match receiver.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(format!("GPU map error: {e}")),
                Err(_) => return Err("readback channel cancelled".to_string()),
            }
            let view = readback_buf.slice(..).get_mapped_range();
            Ok::<Vec<f32>, String>(bytemuck::cast_slice(&view).to_vec())
        });

        result
    }
}
