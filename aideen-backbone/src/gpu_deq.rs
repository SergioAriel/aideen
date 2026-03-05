use aideen_block::cg_bridge::RustCgBridge;
use aideen_block::deq_bridge::{DeqComputeShape, RustDeqBridge};
use aideen_core::state::ArchitectureConfig;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct UpdateUniforms {
    d_model: u32,
    lr: f32,
    grad_scale: f32,
    _pad: u32,
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
    fused_update_params_buf: wgpu::Buffer,
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

        let bridge = RustDeqBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            64,  // max_batch_size (TODO: config)
            256, // max_seq_len (TODO: config)
        );

        let cg_bridge = RustCgBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            64, // max_batch_size
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
            size: 16,
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

    /// Recibe un vector C de matrices `flateadas` y las envía al mega kernel WGSL DEQ.
    /// `b` es batch_size.
    /// Ejecuta el Forward Pass DEQ en WGSL de manera ASÍNCRONA REAL
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
        let shape = DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            _pad: 0,
        };

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

    #[allow(clippy::too_many_arguments)]
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
        let shape = DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            _pad: 0,
        };

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

    #[allow(clippy::too_many_arguments)]
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
        let shape = DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            _pad: 0,
        };

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

    #[allow(clippy::too_many_arguments)]
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
        let shape = DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            _pad: 0,
        };

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

    /// Ejecuta el Backward Pass DEQ (CG Solver) en WGSL de manera ASÍNCRONA REAL
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
        let shape = aideen_block::cg_bridge::CGComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            cg_iters,
            // Más estricto para mejorar exactitud del gradiente implícito.
            epsilon: 5e-4,
        };

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

    #[allow(clippy::too_many_arguments)]
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
        let shape = aideen_block::cg_bridge::CGComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            cg_iters,
            epsilon: 5e-4,
        };

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

    /// Ejecuta el Backward Pass DEQ (CG Solver) sin leer estados a la CPU.
    /// Utiliza el gradiente dL/dh del LM Head trainer directamente en la GPU.
    pub fn run_backward_no_readback(
        &self,
        batch_size: u32,
        _current_seq_tokens: u32, // Para el offset de h_star
        dl_dh_src: &wgpu::Buffer,
        cg_iters: u32,
    ) -> Result<(), String> {
        let shape = aideen_block::cg_bridge::CGComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            cg_iters,
            epsilon: 5e-4,
        };

        // El h_star está en hnext_buf. El offset depende de cuántos tokens hemos procesado.
        // Pero en el forward 'pooled' o 'high_throughput', solemos procesar la secuencia completa.
        // Si es autoregresivo, s_in/h_star se van acumulando.
        // Por ahora asumimos offset 0 para simplificar el MVP de alta velocidad.
        let h_offset = 0;

        self.cg_bridge.run_backward_no_readback(
            &self.device,
            &self.queue,
            &shape,
            &self.bridge.hnext_buf,
            h_offset,
            dl_dh_src,
        )
    }

    /// Aplica la actualización de pesos DEQ de rango-1 directamente en la GPU.
    /// W = W - lr * (v * q^T)
    pub fn apply_fused_deq_update(&self, lr: f32, grad_scale: f32) -> Result<(), String> {
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            lr,
            grad_scale,
            _pad: 0,
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

        // MUY IMPORTANTE: Sincronizar el CG bridge con los nuevos pesos del bridge de forward
        // para que la siguiente iteración use los pesos actualizados.
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
}
