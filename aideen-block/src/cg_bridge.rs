use wgpu::util::DeviceExt;

/// Forma del cómputo para el Conjugate Gradient
#[derive(Debug, Clone)]
pub struct CGComputeShape {
    pub batch_size: u32,
    pub d_model: u32,
    pub h_slots: u32,
    pub cg_iters: u32,
    pub epsilon: f32,
    pub curr_iter: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
}

impl CGComputeShape {
    fn to_uniform(&self) -> [u32; 12] {
        [
            self.batch_size,
            self.d_model,
            self.h_slots,
            self.cg_iters,
            bytemuck::cast(self.epsilon),
            self.curr_iter,
            self._pad0,
            self._pad1,
            self._pad2,
            self._pad3,
            self._pad4,
            self._pad5,
        ]
    }
}

pub struct RustCgBridge {
    pub pipeline: wgpu::ComputePipeline, // legacy – kept for compatibilidad
    pub init_pipeline: wgpu::ComputePipeline, // Kernel 1: inicialización
    pub matvec_pipeline: wgpu::ComputePipeline, // Kernel 2: A*p (sin workgroup memory)
    pub update_pipeline: wgpu::ComputePipeline, // Kernel 3: actualización CG
    pub bind_group_layout_0: wgpu::BindGroupLayout,
    pub bind_group_layout_1: wgpu::BindGroupLayout,

    pub uniform_buf: wgpu::Buffer,
    pub b_sin: wgpu::Buffer,
    pub b_h_star: wgpu::Buffer,
    pub b_dl: wgpu::Buffer,
    pub b_wq: wgpu::Buffer,
    pub b_wk: wgpu::Buffer,
    pub b_wv: wgpu::Buffer,
    pub b_wo: wgpu::Buffer,
    pub b_win: wgpu::Buffer,
    pub b_wx: wgpu::Buffer,
    pub b_wout: wgpu::Buffer,
    pub b_alog: wgpu::Buffer,
    pub b_norm: wgpu::Buffer,
    pub b_v_out: wgpu::Buffer,
    pub b_r: wgpu::Buffer,
    pub b_p: wgpu::Buffer,
    pub b_ap: wgpu::Buffer,
    pub debug_buf: wgpu::Buffer,
    pub b_scalars: wgpu::Buffer, // [f32; batch_size] – rs_old entre iteraciones CG

    pub bind_group_0: wgpu::BindGroup,
    pub bind_group_1: wgpu::BindGroup,
}

impl RustCgBridge {
    pub fn new(device: &wgpu::Device, d_model: u32, h_slots: u32, max_batch_size: u32) -> Self {
        // Shaders: legacy + 3 kernels de multi-dispatch
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN CG Solver Shader (legacy)"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cg_solver.wgsl").into()),
        });
        let shader_init = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CG Init Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cg_init.wgsl").into()),
        });
        let shader_matvec = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CG Matvec Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cg_matvec.wgsl").into()),
        });
        let shader_update = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CG Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cg_update.wgsl").into()),
        });

        // Group 0: Pesos estáticos y estado convergido H*
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CG Bind Group Layout 0 (Pesos y Estado H*)"),
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
                    // Entradas S_in, H_star, dl_dh_pooled
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
                    // W_q a W_out (7 bindings: 4->10)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // A_log y NormScale
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
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

        // Group 1: Buffers iterativos del Algoritmo Conjugate Gradient
        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CG Bind Group Layout 1 (Estado algorítmico RW)"),
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
                    // binding 5: scalars (rs_old/rs_new entre iteraciones)
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
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CG Solver Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CG Solver Compute Pipeline (legacy)"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("cg_solver_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CG Init Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_init,
            entry_point: Some("cg_init_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CG Matvec Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_matvec,
            entry_point: Some("cg_matvec_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CG Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_update,
            entry_point: Some("cg_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ==========================================
        // PERSISTENT GPU MEMORY ALLOCATION
        // (Pre-allocates MAX memory footprint once)
        // ==========================================

        let buf_size = |count: u32| -> wgpu::BufferAddress { (count as wgpu::BufferAddress) * 4 };
        let create_storage = |label: &str, size: wgpu::BufferAddress| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CG Shape Uniform"),
            size: 48, // [u32; 12] = 48 bytes para alineación WGSL vec3
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Group 0 Buffers
        let b_sin = create_storage("S_in", buf_size(max_batch_size * d_model));
        let b_h_star = create_storage("H_star", buf_size(max_batch_size * h_slots * d_model));
        let b_dl = create_storage("dl_dh_pooled", buf_size(max_batch_size * d_model));
        let b_wq = create_storage("W_q", buf_size(d_model * d_model));
        let b_wk = create_storage("W_k", buf_size(d_model * d_model));
        let b_wv = create_storage("W_v", buf_size(d_model * d_model));
        let b_wo = create_storage("W_o", buf_size(d_model * d_model));
        let b_win = create_storage("W_in", buf_size(d_model * d_model));
        let b_wx = create_storage("W_x", buf_size(d_model * d_model));
        let b_wout = create_storage("W_out", buf_size(d_model * d_model));
        let b_alog = create_storage("A_log", buf_size(d_model));
        let b_norm = create_storage("Norm_scale", buf_size(d_model));

        let state_bytes = buf_size(max_batch_size * h_slots * d_model);
        let b_v_out = create_storage("V_out", state_bytes);
        let b_r = create_storage("R (residuos)", state_bytes);
        let b_p = create_storage("P (direcciones)", state_bytes);
        let b_ap = create_storage("AP (producto matriz vector)", state_bytes);
        let debug_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CG Debug Log"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_scalars = create_storage("CG Scalars (rs_old/rs_new)", buf_size(max_batch_size));

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CG Bind Group 0 Persistent"),
            layout: &bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_sin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_h_star.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_dl.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_wq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_wk.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_wv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_wo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_win.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: b_wx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: b_wout.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: b_alog.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: b_norm.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CG Bind Group 1 Persistent"),
            layout: &bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_v_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_ap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        });

        RustCgBridge {
            pipeline,
            init_pipeline,
            matvec_pipeline,
            update_pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            uniform_buf,
            b_sin,
            b_h_star,
            b_dl,
            b_wq,
            b_wk,
            b_wv,
            b_wo,
            b_win,
            b_wx,
            b_wout,
            b_alog,
            b_norm,
            b_v_out,
            b_r,
            b_p,
            b_ap,
            debug_buf,
            b_scalars,
            bind_group_0,
            bind_group_1,
        }
    }

    /// Despacha los 3 kernels CG en multi-pass:
    ///   1 encoder  → init (una vez)
    ///   1 encoder  → cg_iters × (matvec + update)
    fn dispatch_cg_multi_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: u32,
        cg_iters: u32,
        shape: &CGComputeShape,
    ) {
        // --- Init (encoder propio) ---
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Init Encoder"),
        });
        {
            let mut s = shape.clone();
            s.curr_iter = 0;
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&s.to_uniform()));

            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CG Init Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.init_pipeline);
            cpass.set_bind_group(0, &self.bind_group_0, &[]);
            cpass.set_bind_group(1, &self.bind_group_1, &[]);
            cpass.dispatch_workgroups(batch_size, 1, 1);
        }
        queue.submit(Some(enc.finish()));

        // --- Iteraciones CG ---
        for i in 0..cg_iters {
            let mut s = shape.clone();
            s.curr_iter = i;
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&s.to_uniform()));

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(format!("CG Iteration {i} Encoder").as_str()),
            });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG Matvec Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.matvec_pipeline);
                cpass.set_bind_group(0, &self.bind_group_0, &[]);
                cpass.set_bind_group(1, &self.bind_group_1, &[]);
                cpass.dispatch_workgroups(batch_size, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG Update Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.update_pipeline);
                cpass.set_bind_group(0, &self.bind_group_0, &[]);
                cpass.set_bind_group(1, &self.bind_group_1, &[]);
                cpass.dispatch_workgroups(batch_size, 1, 1);
            }
            queue.submit(Some(enc.finish()));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_backward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &CGComputeShape,
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
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        assert!(
            shape.batch_size <= 64,
            "Batches over 64 require remaking the bridge currently"
        );

        // 1) Actualizar Memoria Persistente (No Allocations here!)
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&shape.to_uniform()),
        );
        queue.write_buffer(&self.b_sin, 0, bytemuck::cast_slice(s_in));
        queue.write_buffer(&self.b_h_star, 0, bytemuck::cast_slice(h_star));
        queue.write_buffer(&self.b_dl, 0, bytemuck::cast_slice(dl_dh_pooled));

        if update_weights {
            queue.write_buffer(&self.b_wq, 0, bytemuck::cast_slice(w_q));
            queue.write_buffer(&self.b_wk, 0, bytemuck::cast_slice(w_k));
            queue.write_buffer(&self.b_wv, 0, bytemuck::cast_slice(w_v));
            queue.write_buffer(&self.b_wo, 0, bytemuck::cast_slice(w_o));
            queue.write_buffer(&self.b_win, 0, bytemuck::cast_slice(w_in));
            queue.write_buffer(&self.b_wx, 0, bytemuck::cast_slice(w_x));
            queue.write_buffer(&self.b_wout, 0, bytemuck::cast_slice(w_out));
            queue.write_buffer(&self.b_alog, 0, bytemuck::cast_slice(a_log));
            queue.write_buffer(&self.b_norm, 0, bytemuck::cast_slice(norm));
        }
        self.dispatch_cg_multi_pass(device, queue, shape.batch_size, shape.cg_iters, shape);

        let readback_bytes = (shape.batch_size * shape.d_model * 4) as wgpu::BufferAddress;
        let staging_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer V_out"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.b_v_out, 0, &staging_v, 0, readback_bytes);
        queue.submit(Some(encoder.finish()));

        // Mapear VRAM
        let slice_v = staging_v.slice(..);

        let done = std::sync::Arc::new(std::sync::Mutex::new(false));
        let done_ref = done.clone();

        slice_v.map_async(wgpu::MapMode::Read, move |v| {
            if v.is_ok() {
                *done_ref.lock().unwrap() = true;
            }
        });

        // Loop de Poll para Single Threaded Executors
        loop {
            device.poll(wgpu::Maintain::Wait);
            if *done.lock().unwrap() {
                break;
            }
            std::thread::yield_now(); // <-- Crucial
        }

        let v_data: Vec<f32> = {
            let mapping = slice_v.get_mapped_range();
            bytemuck::cast_slice(&mapping).to_vec()
        };

        staging_v.unmap();
        Ok(v_data)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sync_weights_from_deq_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wq_src: &wgpu::Buffer,
        wk_src: &wgpu::Buffer,
        wv_src: &wgpu::Buffer,
        wo_src: &wgpu::Buffer,
        win_src: &wgpu::Buffer,
        wx_src: &wgpu::Buffer,
        wout_src: &wgpu::Buffer,
        a_src: &wgpu::Buffer,
        n_src: &wgpu::Buffer,
        d_model: u32,
    ) {
        let mat_bytes = (d_model * d_model * 4) as wgpu::BufferAddress;
        let vec_bytes = (d_model * 4) as wgpu::BufferAddress;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Sync Weights From DEQ"),
        });
        encoder.copy_buffer_to_buffer(wq_src, 0, &self.b_wq, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(wk_src, 0, &self.b_wk, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(wv_src, 0, &self.b_wv, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(wo_src, 0, &self.b_wo, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(win_src, 0, &self.b_win, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(wx_src, 0, &self.b_wx, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(wout_src, 0, &self.b_wout, 0, mat_bytes);
        encoder.copy_buffer_to_buffer(a_src, 0, &self.b_alog, 0, vec_bytes);
        encoder.copy_buffer_to_buffer(n_src, 0, &self.b_norm, 0, vec_bytes);
        queue.submit(Some(encoder.finish()));
    }

    pub fn run_backward_from_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &CGComputeShape,
        s_in: &[f32],
        h_star_src: &wgpu::Buffer,
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
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        assert!(
            shape.batch_size <= 64,
            "Batches over 64 require remaking the bridge currently"
        );

        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&shape.to_uniform()),
        );
        queue.write_buffer(&self.b_sin, 0, bytemuck::cast_slice(s_in));
        queue.write_buffer(&self.b_dl, 0, bytemuck::cast_slice(dl_dh_pooled));

        if update_weights {
            queue.write_buffer(&self.b_wq, 0, bytemuck::cast_slice(w_q));
            queue.write_buffer(&self.b_wk, 0, bytemuck::cast_slice(w_k));
            queue.write_buffer(&self.b_wv, 0, bytemuck::cast_slice(w_v));
            queue.write_buffer(&self.b_wo, 0, bytemuck::cast_slice(w_o));
            queue.write_buffer(&self.b_win, 0, bytemuck::cast_slice(w_in));
            queue.write_buffer(&self.b_wx, 0, bytemuck::cast_slice(w_x));
            queue.write_buffer(&self.b_wout, 0, bytemuck::cast_slice(w_out));
            queue.write_buffer(&self.b_alog, 0, bytemuck::cast_slice(a_log));
            queue.write_buffer(&self.b_norm, 0, bytemuck::cast_slice(norm));
        }

        let state_bytes =
            (shape.batch_size * shape.h_slots * shape.d_model * 4) as wgpu::BufferAddress;
        let readback_bytes = (shape.batch_size * shape.d_model * 4) as wgpu::BufferAddress;

        // Copia GPU→GPU de H* antes del multi-dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG H_star Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(h_star_src, h_offset, &self.b_h_star, 0, state_bytes);
        queue.submit(Some(encoder.finish()));
        self.dispatch_cg_multi_pass(device, queue, shape.batch_size, shape.cg_iters, shape);

        let staging_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer V_out"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.b_v_out, 0, &staging_v, 0, readback_bytes);
        queue.submit(Some(encoder.finish()));

        let slice_v = staging_v.slice(..);
        let done = std::sync::Arc::new(std::sync::Mutex::new(false));
        let done_ref = done.clone();
        slice_v.map_async(wgpu::MapMode::Read, move |v| {
            if v.is_ok() {
                *done_ref.lock().unwrap() = true;
            }
        });

        loop {
            device.poll(wgpu::Maintain::Wait);
            if *done.lock().unwrap() {
                break;
            }
            std::thread::yield_now();
        }

        let v_data: Vec<f32> = {
            let mapping = slice_v.get_mapped_range();
            bytemuck::cast_slice(&mapping).to_vec()
        };

        staging_v.unmap();
        Ok(v_data)
    }

    /// Ejecuta el CG Solver usando el gradiente dl_dh ya presente en la GPU (de GpuLmHeadTrainer).
    pub fn run_backward_no_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &CGComputeShape,
        h_star_src: &wgpu::Buffer,
        h_offset: u64,
        dl_dh_src: &wgpu::Buffer,
    ) -> Result<(), String> {
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&shape.to_uniform()),
        );

        // Copiar gradiente dL/dh calculado por el LM Head directo al input del CG
        // dl_size debe ser batch_size * d_model * 4
        let dl_size = (shape.batch_size as u64) * (shape.d_model as u64) * 4;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Solver Encoder (Zero Readback)"),
        });

        let state_bytes =
            (shape.batch_size * shape.h_slots * shape.d_model * 4) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(dl_dh_src, 0, &self.b_dl, 0, dl_size);
        encoder.copy_buffer_to_buffer(h_star_src, h_offset, &self.b_h_star, 0, state_bytes);
        queue.submit(Some(encoder.finish()));

        self.dispatch_cg_multi_pass(device, queue, shape.batch_size, shape.cg_iters, shape);
        Ok(())
    }

    pub fn read_debug_buffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let size = self.debug_buf.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CG Debug Staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG Debug Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.debug_buf, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            data
        } else {
            vec![]
        }
    }
}
