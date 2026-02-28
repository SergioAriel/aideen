// dispatch.rs
// ─────────────────────────────────────────────────────────────────────────────
// Kernel dispatch: compiles WGSL shaders and executes them on the GPU.
//
// Architecture:
//   KernelCache  — lazily compiles each WGSL shader once, caches pipelines
//   Dispatcher   — high-level API: matmul(a, b) → c
//
// Shader source code is embedded at compile time via include_str!.
// This means the .wgsl files are baked into the binary — no file I/O at runtime.
//
// Dispatch pattern:
//   1. Bind input/output buffers to bind groups
//   2. Encode dispatch in a command encoder
//   3. Submit to queue
//   4. (Sync) poll device or use async fence
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline, Device, Queue, ShaderModule};

use crate::tensor::{GpuContext, Shape, Tensor};

// ─── Embedded shaders ─────────────────────────────────────────────────────────

const MATMUL_WGSL: &str = include_str!("kernels/matmul.wgsl");
const LAYERNORM_WGSL: &str = include_str!("kernels/layernorm.wgsl");
const SOFTMAX_WGSL: &str = include_str!("kernels/softmax.wgsl");
const SILU_WGSL: &str = include_str!("kernels/silu.wgsl");
const ADAMW_WGSL: &str = include_str!("kernels/adamw.wgsl");
const ROPE_WGSL: &str = include_str!("kernels/rope.wgsl");
const CD_DYNAMICS_WGSL: &str = include_str!("kernels/cd_dynamics.wgsl");

// ─── Uniform structs (must match WGSL layout exactly) ─────────────────────────

/// Must be repr(C) + Pod to safely cast to &[u8] for GPU upload.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatMulDims {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LayerNormParams {
    pub n_rows: u32,
    pub d_model: u32,
    pub eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SoftmaxParams {
    pub n_rows: u32,
    pub d_logits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ElemParams {
    pub n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AdamWParams {
    pub n: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub bc1: f32, // 1 - beta1^t
    pub bc2: f32, // 1 - beta2^t
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RopeParams {
    pub batch: u32,
    pub heads: u32,
    pub seq: u32,
    pub d_head: u32,
    pub base: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CdUpdateParams {
    pub n: u32,
    pub lambda_c: f32,
    pub alpha_r: f32,
    pub eps_norm: f32,
}

// ─── Kernel ID ────────────────────────────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum KernelId {
    MatMul,
    LayerNorm,
    Softmax,
    SiluForward,
    GeluForward,
    AddForward,
    MulForward,
    AdamW,
    RoPE,
    CdMeanDelta,
    CdApplyUpdate,
    CdAccumRepulsion,
}

// ─── Compiled Pipeline Entry ─────────────────────────────────────────────────

struct CompiledPipeline {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
}

// ─── Kernel Cache ─────────────────────────────────────────────────────────────

/// Lazily compiles and caches compute pipelines.
/// Thread-safe via Mutex — pipelines are compiled once.
pub struct KernelCache {
    ctx: GpuContext,
    cache: Mutex<HashMap<KernelId, CompiledPipeline>>,
}

impl KernelCache {
    pub fn new(ctx: GpuContext) -> Self {
        Self {
            ctx,
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn compile_shader(&self, source: &str, label: &str) -> ShaderModule {
        self.ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    fn get_or_compile(&self, id: KernelId) -> Arc<ComputePipeline> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get(&id) {
            return Arc::clone(&entry.pipeline);
        }

        let (shader_src, entry_point, label) = match &id {
            KernelId::MatMul => (MATMUL_WGSL, "main", "matmul"),
            KernelId::LayerNorm => (LAYERNORM_WGSL, "main", "layernorm"),
            KernelId::Softmax => (SOFTMAX_WGSL, "main", "softmax"),
            KernelId::SiluForward => (SILU_WGSL, "silu_forward", "silu"),
            KernelId::GeluForward => (SILU_WGSL, "gelu_forward", "gelu"),
            KernelId::AddForward => (SILU_WGSL, "add_forward", "add"),
            KernelId::MulForward => (SILU_WGSL, "mul_forward", "mul"),
            KernelId::AdamW => (ADAMW_WGSL, "main", "adamw"),
            KernelId::RoPE => (ROPE_WGSL, "main", "rope"),
            KernelId::CdMeanDelta => (CD_DYNAMICS_WGSL, "compute_mean_delta", "cd_mean"),
            KernelId::CdApplyUpdate => (CD_DYNAMICS_WGSL, "apply_cd_gradient", "cd_apply"),
            KernelId::CdAccumRepulsion => (CD_DYNAMICS_WGSL, "accumulate_repulsion", "cd_repul"),
        };

        let shader = self.compile_shader(shader_src, label);
        let pipeline = Arc::new(self.ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // auto layout from shader reflection
                module: &shader,
                entry_point,
                compilation_options: Default::default(),
                // cache: None,
            },
        ));

        let bind_group_layout = Arc::new(pipeline.get_bind_group_layout(0));

        cache.insert(
            id.clone(),
            CompiledPipeline {
                pipeline: Arc::clone(&pipeline),
                bind_group_layout: Arc::clone(&bind_group_layout),
            },
        );

        pipeline
    }
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────

/// High-level dispatch API.
/// Each method takes Tensor inputs and returns a Tensor output.
/// All operations are async-compatible — they encode but don't block
/// unless you call ctx.device.poll(Wait).
pub struct Dispatcher {
    pub ctx: GpuContext,
    pub cache: Arc<KernelCache>,
}

impl Dispatcher {
    pub fn new(ctx: GpuContext) -> Self {
        let cache = Arc::new(KernelCache::new(ctx.clone()));
        Self { ctx, cache }
    }

    /// Helper: create a uniform buffer from a Pod type.
    fn uniform_buf<T: Pod>(&self, data: &T) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniform"),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Dispatch a compute pass and submit to queue.
    /// dispatch_xy: (x_groups, y_groups)
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        dispatch_x: u32,
        dispatch_y: u32,
        label: &str,
    ) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.ctx.queue.submit([encoder.finish()]);
    }

    // ── Operations ────────────────────────────────────────────────────────────

    /// Matrix multiplication: C = A × B
    /// A: [M, K], B: [K, N] → C: [M, N]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape = a.shape.check_matmul(&b.shape)?;
        let (m, k, n) = (
            a.shape.dim(0)? as u32,
            a.shape.dim(1)? as u32,
            b.shape.dim(1)? as u32,
        );

        let output = Tensor::uninit(out_shape, self.ctx.clone());
        let dims_buf = self.uniform_buf(&MatMulDims { m, k, n, _pad: 0 });

        let pipeline = self.cache.get_or_compile(KernelId::MatMul);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });

        // Tile size is 16×16
        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);
        self.dispatch(&pipeline, &bind_group, gx, gy, "matmul");

        Ok(output)
    }

    /// Layer normalization: y = (x - mean) / sqrt(var + ε) * γ + β
    /// x: [N, D], gamma: [D], beta: [D] → y: [N, D]
    pub fn layernorm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
        if x.shape.ndim() != 2 {
            return Err(anyhow!("layernorm expects 2D input [N, D]"));
        }
        let (n, d) = (x.shape.dim(0)? as u32, x.shape.dim(1)? as u32);

        let output = Tensor::uninit(x.shape.clone(), self.ctx.clone());
        let params_buf = self.uniform_buf(&LayerNormParams {
            n_rows: n,
            d_model: d,
            eps,
            _pad: 0,
        });

        let pipeline = self.cache.get_or_compile(KernelId::LayerNorm);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layernorm_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gamma.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: beta.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        // One workgroup per row
        self.dispatch(&pipeline, &bind_group, n, 1, "layernorm");
        Ok(output)
    }

    /// Softmax over last dimension: y = softmax(x, dim=-1)
    /// x: [N, D] → y: [N, D]
    pub fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        if x.shape.ndim() != 2 {
            return Err(anyhow!("softmax expects 2D input [N, D]"));
        }
        let (n, d) = (x.shape.dim(0)? as u32, x.shape.dim(1)? as u32);

        let output = Tensor::uninit(x.shape.clone(), self.ctx.clone());
        let params_buf = self.uniform_buf(&SoftmaxParams {
            n_rows: n,
            d_logits: d,
        });

        let pipeline = self.cache.get_or_compile(KernelId::Softmax);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, n, 1, "softmax");
        Ok(output)
    }

    /// SiLU activation: y = x * sigmoid(x)
    pub fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.numel() as u32;
        let output = Tensor::uninit(x.shape.clone(), self.ctx.clone());
        let params_buf = self.uniform_buf(&ElemParams { n });

        let pipeline = self.cache.get_or_compile(KernelId::SiluForward);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("silu_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, n.div_ceil(256), 1, "silu");
        Ok(output)
    }

    /// Elementwise add: C = A + B (same shape required)
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.shape.check_broadcast(&b.shape)?;
        let n = a.numel() as u32;
        let output = Tensor::uninit(a.shape.clone(), self.ctx.clone());
        let params_buf = self.uniform_buf(&ElemParams { n });

        let pipeline = self.cache.get_or_compile(KernelId::AddForward);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, n.div_ceil(256), 1, "add");
        Ok(output)
    }

    /// AdamW optimizer step — updates theta in-place.
    pub fn adamw_step(
        &self,
        theta: &Tensor,
        grad: &Tensor,
        moment1: &Tensor,
        moment2: &Tensor,
        params: AdamWParams,
    ) -> Result<()> {
        let params_buf = self.uniform_buf(&params);
        let n = theta.numel() as u32;

        let pipeline = self.cache.get_or_compile(KernelId::AdamW);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adamw_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: theta.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: moment1.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: moment2.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, n.div_ceil(256), 1, "adamw");
        Ok(())
    }

    /// RoPE encoding for Q or K tensor.
    pub fn rope(
        &self,
        x: &Tensor,
        batch: u32,
        heads: u32,
        seq: u32,
        d_head: u32,
        base: f32,
    ) -> Result<Tensor> {
        let output = Tensor::uninit(x.shape.clone(), self.ctx.clone());
        let params_buf = self.uniform_buf(&RopeParams {
            batch,
            heads,
            seq,
            d_head,
            base,
            _pad: 0,
        });
        let total_pairs = batch * heads * seq * (d_head / 2);

        let pipeline = self.cache.get_or_compile(KernelId::RoPE);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rope_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, total_pairs.div_ceil(64), 1, "rope");
        Ok(output)
    }

    /// Apply C+D gradient to expert weights in-place.
    pub fn cd_apply_update(
        &self,
        theta: &Tensor,
        mean_delta: &Tensor,
        repulsion_force: &Tensor,
        lambda_c: f32,
        alpha_r: f32,
    ) -> Result<()> {
        let n = theta.numel() as u32;
        let params_buf = self.uniform_buf(&CdUpdateParams {
            n,
            lambda_c,
            alpha_r,
            eps_norm: 5.0,
        });

        let pipeline = self.cache.get_or_compile(KernelId::CdApplyUpdate);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cd_apply_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: theta.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mean_delta.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: repulsion_force.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        self.dispatch(&pipeline, &bind_group, n.div_ceil(256), 1, "cd_apply");
        Ok(())
    }
}
