use aideen_block::cg_bridge::RustCgBridge;
use aideen_block::deq_bridge::{DeqComputeShape, RustDeqBridge};
use aideen_core::state::ArchitectureConfig;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct UpdateUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    _pad2: u32,
}

/// Abstracción del DEQ vía GPU (WGPU).
pub struct GpuDeqBackend {
    pub config: ArchitectureConfig,
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub bridge: RustDeqBridge,
    pub cg_bridge: RustCgBridge,

    // Fused update pipeline
    fused_update_pipeline: wgpu::ComputePipeline,
    fused_update_bg0: wgpu::BindGroup,
    fused_update_bg1: wgpu::BindGroup,
    pub fused_update_params_buf: wgpu::Buffer,
}

impl GpuDeqBackend {
    /// Inicializa la conexión con Apple Metal / Vulkan y compila los Shaders WGSL del DEQ
    pub async fn new_async(config: ArchitectureConfig) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        println!("[GpuDeqBackend] Adapter: {}", adapter.get_info().name);
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = 16;

        // 3. Crear Device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("AIDEEN DEQ GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let safe_batch = config.ctx_len.max(1024);
        let safe_seq = config.ctx_len.max(256); // DEQ causal usa 256 típico

        // RustDeqBridge::new(device, d_model, h_slots, max_batch_size, max_seq_len)
        let bridge = RustDeqBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            safe_batch as u32, // ✅ max_batch_size
            safe_seq as u32,   // ✅ max_seq_len
        );

        let cg_bridge = RustCgBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            safe_batch as u32,
        );

        // Fused Update Pipeline setup
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused DEQ Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fused_deq_update.wgsl").into()),
        });

        let bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Update BG0 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let bg1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Update BG1 Layout (Weights)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
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

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused Update PL"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });

        let fused_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_deq_update_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let fused_update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Update Params"),
            size: 32, // Match UpdateUniforms (32 bytes: 6 fields + 2 padding = 8 u32/f32)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fused_update_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Update BG0"),
            layout: &bg0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fused_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cg_bridge.b_v_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.debug_buf.as_entire_binding(),
                },
            ],
        });

        let fused_update_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Update BG1"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bridge.wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bridge.wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.wo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bridge.wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bridge.a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bridge.n_buf.as_entire_binding(),
                },
            ],
        });

        Some(Self {
            config,
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bridge,
            cg_bridge,
            fused_update_pipeline,
            fused_update_bg0,
            fused_update_bg1,
            fused_update_params_buf,
        })
    }

    /// Método sincrónico por si necesitamos llamarlo desde el main hilo (pollster) V1.
    pub fn new_blocking(config: ArchitectureConfig) -> Option<Self> {
        pollster::block_on(Self::new_async(config))
    }

    /// Reinicia los estados ocultos (slots) en la GPU a cero.
    pub fn reset_state(&self) {
        let size = self.bridge.hcurr_buf.size();
        let zeros = vec![0.0f32; (size / 4) as usize];
        self.queue
            .write_buffer(&self.bridge.hcurr_buf, 0, bytemuck::cast_slice(&zeros));
        self.queue
            .write_buffer(&self.bridge.hnext_buf, 0, bytemuck::cast_slice(&zeros));
    }

    // --- HELPER UNIFICADO ---
    fn build_compute_shape(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
    ) -> DeqComputeShape {
        DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            _pad: 0,
        }
    }

    pub fn build_cg_shape(
        &self,
        batch_size: u32,
        cg_iters: u32,
    ) -> aideen_block::cg_bridge::CGComputeShape {
        aideen_block::cg_bridge::CGComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            cg_iters,
            epsilon: self.config.deq_epsilon,
            curr_iter: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
        }
    }

    // --- WEIGHTS ---

    pub fn upload_weights(
        &self,
        queue: &wgpu::Queue,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        win: &[f32],
        wx: &[f32],
        wout: &[f32],
        alog: &[f32],
        nscale: &[f32],
    ) {
        queue.write_buffer(&self.bridge.wq_buf, 0, bytemuck::cast_slice(wq));
        queue.write_buffer(&self.bridge.wk_buf, 0, bytemuck::cast_slice(wk));
        queue.write_buffer(&self.bridge.wv_buf, 0, bytemuck::cast_slice(wv));
        queue.write_buffer(&self.bridge.wo_buf, 0, bytemuck::cast_slice(wo));
        queue.write_buffer(&self.bridge.win_buf, 0, bytemuck::cast_slice(win));
        queue.write_buffer(&self.bridge.wx_buf, 0, bytemuck::cast_slice(wx));
        queue.write_buffer(&self.bridge.wout_buf, 0, bytemuck::cast_slice(wout));
        queue.write_buffer(&self.bridge.a_buf, 0, bytemuck::cast_slice(alog));
        queue.write_buffer(&self.bridge.n_buf, 0, bytemuck::cast_slice(nscale));
        queue.submit([]);
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok((wq_check, _, _, _, win_check, _, _, _, _)) = self.read_weights() {
            let stats = |v: &[f32]| -> (f32, f32, f32) {
                let min = v.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let abs_mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                (min, max, abs_mean)
            };
            let (q_min, q_max, q_abs) = stats(&wq_check);
            let (in_min, in_max, in_abs) = stats(&win_check);
            eprintln!(
                "[GPU-VERIFY] Post-upload:\n    W_q:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_in: min={:.4}, max={:.4}, abs_mean={:.6}",
                q_min, q_max, q_abs, in_min, in_max, in_abs
            );
        }
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
    }

    // --- MÉTODOS DE EJECUCIÓN ---

    pub fn run_forward_deq(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    // ESTE ES EL QUE ESPERA EL TRAINER (16 argumentos)
    pub fn run_forward_deq_pooled(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_pooled(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_single_token(
        &self,
        batch_size: u32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool, // <--- ARREGLADO: Agregado el argumento faltante
    ) -> Result<Vec<f32>, &'static str> {
        let shape =
            self.build_compute_shape(batch_size, 1, self.config.max_deq_iters as u32, 5e-4, 0.9);

        // ARREGLADO: Extraemos el primer elemento de la tupla (Vec<f32>, Vec<u32>)
        match self.bridge.run_forward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        ) {
            Ok((h_data, _)) => Ok(h_data),
            Err(e) => Err(e),
        }
    }

    pub fn run_forward_deq_pooled_with_state(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_pooled_with_state(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_deq_no_readback(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_no_readback(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_from_seq_buf(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_buf_gpu: &wgpu::Buffer,
        update_weights: bool,
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
    ) -> Result<(), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.queue
            .write_buffer(&self.bridge.uniform_buf, 0, bytemuck::bytes_of(&shape));

        if update_weights {
            self.queue
                .write_buffer(&self.bridge.wq_buf, 0, bytemuck::cast_slice(w_q));
            self.queue
                .write_buffer(&self.bridge.wk_buf, 0, bytemuck::cast_slice(w_k));
            self.queue
                .write_buffer(&self.bridge.wv_buf, 0, bytemuck::cast_slice(w_v));
            self.queue
                .write_buffer(&self.bridge.wo_buf, 0, bytemuck::cast_slice(w_o));
            self.queue
                .write_buffer(&self.bridge.win_buf, 0, bytemuck::cast_slice(w_in));
            self.queue
                .write_buffer(&self.bridge.wx_buf, 0, bytemuck::cast_slice(w_x));
            self.queue
                .write_buffer(&self.bridge.wout_buf, 0, bytemuck::cast_slice(w_out));
            self.queue
                .write_buffer(&self.bridge.a_buf, 0, bytemuck::cast_slice(a_log));
            self.queue
                .write_buffer(&self.bridge.n_buf, 0, bytemuck::cast_slice(norm));
        }

        let copy_size = (batch_size as u64) * (seq_len as u64) * (self.config.d_r as u64) * 4;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DEQ Forward (GPU s_buf) Encoder"),
            });
        encoder.copy_buffer_to_buffer(s_buf_gpu, 0, &self.bridge.s_buf, 0, copy_size);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward Pass (GPU s_buf)"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bridge.pipeline);
            cpass.set_bind_group(0, &self.bridge.bind_group, &[]);
            cpass.dispatch_workgroups(batch_size.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    // --- BACKWARD ---

    pub fn run_backward_deq(
        &self,
        batch_size: u32,
        s_in: &[f32],
        h_star: &[f32],
        dl_dh_pooled: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        cg_iters: u32,
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        self.cg_bridge.run_backward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            h_star,
            dl_dh_pooled,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_backward_deq_from_forward_state(
        &self,
        batch_size: u32,
        s_in: &[f32],
        h_offset: u64,
        dl_dh_pooled: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        cg_iters: u32,
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        self.cg_bridge.run_backward_from_buffer(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            &self.bridge.hnext_buf,
            h_offset,
            dl_dh_pooled,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn apply_deq_sgd_update_and_sync_cg(
        &self,
        lr: f32,
        grad_mat: &[f32],
        grad_vec: &[f32],
    ) -> Result<(), String> {
        self.bridge
            .apply_sgd_update(&self.device, &self.queue, lr, grad_mat, grad_vec)
            .map_err(|e| e.to_string())?;
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    pub fn run_backward_no_readback(
        &self,
        batch_size: u32,
        _num_tokens: u32,
        dl_dh_src: &wgpu::Buffer,
        cg_iters: u32,
    ) -> Result<(), String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        let h_offset = 0;
        self.cg_bridge.run_backward_no_readback(
            &self.device,
            &self.queue,
            &shape,
            &self.bridge.hcurr_buf,
            h_offset,
            dl_dh_src,
        )
    }

    pub fn apply_fused_deq_update(
        &self,
        lr: f32,
        grad_scale: f32,
        ternary: bool,
        weight_decay: f32,
        seq_len: u32,
    ) -> Result<(), String> {
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            lr,
            grad_scale,
            ternary_flag: if ternary { 1 } else { 0 },
            weight_decay,
            seq_len,
            _pad2: 0,
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fused DEQ Update Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fused DEQ Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fused_update_pipeline);
            pass.set_bind_group(0, &self.fused_update_bg0, &[]);
            pass.set_bind_group(1, &self.fused_update_bg1, &[]);
            let d = self.config.d_r as u32;
            pass.dispatch_workgroups(d.div_ceil(16), d.div_ceil(16), 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    /// Spectral renormalization fully on GPU: power iteration on W_q..W_out,
    /// then syncs updated weights to the CG bridge.
    pub fn renormalize_spectral(&self) -> Result<(), String> {
        self.bridge.renormalize_spectral(
            &self.device,
            &self.queue,
            self.config.d_r as u32,
            0.8, // threshold (same as CPU version)
            10,  // n_iters (same as CPU version)
        );
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    pub fn read_weights(
        &self,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        &'static str,
    > {
        self.bridge
            .read_weights(&self.device, &self.queue, self.config.d_r as u32)
    }

    pub fn read_debug_buffer(&self) -> Vec<f32> {
        self.bridge.read_debug_buffer(&self.device, &self.queue)
    }

    pub fn read_scratch_buffer(&self) -> Vec<f32> {
        self.bridge.read_scratch_buffer(&self.device, &self.queue)
    }

    pub fn read_cg_debug_buffer(&self) -> Vec<f32> {
        self.cg_bridge.read_debug_buffer(&self.device, &self.queue)
    }

    pub fn run_forward(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        damping: f32,
    ) -> Result<(), &'static str> {
        let shape = aideen_block::deq_bridge::DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32, // ✅ consistencia total con el pipeline
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon: self.config.deq_epsilon,
            damping,
            seq_len,
            _pad: 0,
        };

        self.bridge
            .run_forward_gpu_only(&self.device, &self.queue, &shape);
        Ok(())
    }
}
