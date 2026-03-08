use futures::channel::oneshot;
/// Aideen V8 — Forward Pass
///
/// Engineering principles:
///
/// 1. ASYNC READBACK: Uses `futures::channel::oneshot` for the GPU buffer mapping.
///    The map_async callback sends on a oneshot::Sender<T> (which is Send + 'static).
///    Awaiting the Receiver yields correctly to the WASM event loop without spin-loops,
///    timeouts, or AtomicBool polling. Chrome fires the callback as a microtask once
///    the GPU work is complete — the await resumes one tick later.
///
/// 2. STUB PATTERN: All unimplemented passes (Mamba, Attention, MoE, Projection)
///    use identity passthrough: they return the input buffer unchanged.
///    → Zero tensors never enter the pipeline.
///    → Real embedding values flow through to lm_head, producing meaningful results.
///    → Replacing a stub with a real implementation is a one-line change.
///
/// 3. DIMENSION DETECTION: lm_head detects the hidden dimension D from buffer.size()
///    at runtime, not from any config constant. This makes it robust to the model
///    variant actually loaded (e.g. aideen_v7 with D=256 vs aideen_v8 with D=256→4096).
///
/// 4. CROSS-PLATFORM: BlockError is BlockError on wasm32, String on native.
///    Use the `block_err!(msg)` macro consistently to create errors.
use std::collections::HashMap;
use std::sync::Arc;
use wgpu;
use wgpu::util::DeviceExt;

// ── Platform-agnostic error type ──────────────────────────────────────────────
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
type BlockError = wasm_bindgen::BlockError;

#[cfg(not(target_arch = "wasm32"))]
type BlockError = String;

/// Create a BlockError from a string literal (works on both WASM and native).
macro_rules! block_err {
    ($msg:expr) => {{
        #[cfg(target_arch = "wasm32")]
        { wasm_bindgen::block_err!($msg) }
        #[cfg(not(target_arch = "wasm32"))]
        { $msg.to_string() }
    }};
    ($fmt:literal, $($arg:expr),+) => {{
        let s = format!($fmt, $($arg),+);
        #[cfg(target_arch = "wasm32")]
        { wasm_bindgen::block_err!(&s) }
        #[cfg(not(target_arch = "wasm32"))]
        { s }
    }};
}

use super::config::AideenConfig;
use super::layer::{build_layer_stack, AideenLayer};

pub struct AideenForwardPass {
    pub config: AideenConfig,
    pub layers: Vec<AideenLayer>,
}

impl AideenForwardPass {
    pub fn new(config: AideenConfig) -> Self {
        let layers = build_layer_stack(&config);
        Self { config, layers }
    }

    /// Full async forward pass.
    /// Flow: token_ids → embedding → [mamba → attn → moe] × N layers → lm_head → argmax token
    pub async fn run_async(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        token_ids: &[u32],
        weight_buffers: &HashMap<String, wgpu::Buffer>,
    ) -> Result<f32, BlockError> {
        if token_ids.is_empty() {
            return Err(block_err!("forward: empty token sequence"));
        }
        let seq_len = token_ids.len();
        let hidden = self.embedding_lookup(device, queue, token_ids, weight_buffers)?;
        let hidden = self
            .run_layers(device, queue, hidden, weight_buffers, seq_len)
            .await?;
        self.lm_head(device, queue, hidden, weight_buffers, seq_len)
            .await
    }

    // ── Embedding Lookup ─────────────────────────────────────────────────────
    // Dispatches embedding.wgsl: gathers the embedding row for each token.
    // Output: [seq_len × D] f32 buffer on GPU.

    fn embedding_lookup(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        token_ids: &[u32],
        weight_buffers: &HashMap<String, wgpu::Buffer>,
    ) -> Result<wgpu::Buffer, BlockError> {
        let d = self.layers.first().map(|l| l.d_in).unwrap_or(256);
        let t = token_ids.len();

        let emb_buf = weight_buffers
            .get("embedding.weight")
            .ok_or_else(|| block_err!("embedding.weight not found in VRAM"))?;

        let shape_data: [u32; 4] = [t as u32, d as u32, self.config.vocab_size as u32, 0];
        let shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("embed_shape"),
            contents: bytemuck::cast_slice(&shape_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ids_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("token_ids"),
            contents: bytemuck::cast_slice(token_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed_out"),
            size: (t * d * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding_gather"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/embedding.wgsl").into()),
        });
        let bgl = Self::make_bgl(
            device,
            "embed_bgl",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let pipeline = Self::make_pipeline(device, "embed_pipe", &bgl, &shader, "embedding_gather");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: emb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embed_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(t as u32, ((d + 63) / 64) as u32, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        Ok(out_buf)
    }

    // ── Layer Stack ───────────────────────────────────────────────────────────

    async fn run_layers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        mut hidden: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        seq_len: usize,
    ) -> Result<wgpu::Buffer, BlockError> {
        for layer in &self.layers {
            // Apply inter-block projection BEFORE the layer if dimension grows
            if let Some((d_from, d_to)) = self.config.inter_block_projection(layer.idx) {
                hidden = self.project_dimensions(
                    device,
                    queue,
                    hidden,
                    weight_buffers,
                    layer.idx,
                    d_from,
                    d_to,
                    seq_len,
                )?;
            }
            hidden = self
                .run_single_layer(device, queue, hidden, weight_buffers, layer, seq_len)
                .await?;
        }
        Ok(hidden)
    }

    async fn run_single_layer(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        hidden: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        layer: &AideenLayer,
        seq_len: usize,
    ) -> Result<wgpu::Buffer, BlockError> {
        // Layers are residual within blocks. Inter-block projections are handled in run_layers.

        let hidden = self.mamba_pass(device, queue, hidden, weight_buffers, layer)?;

        let hidden = if layer.attn_dims.is_some() {
            self.attention_pass(device, queue, hidden, weight_buffers, layer, seq_len)?
        } else {
            hidden
        };

        if layer.has_moe {
            self.moe_pass(device, queue, hidden, weight_buffers, layer, seq_len)
                .await
        } else {
            Ok(hidden)
        }
    }

    // ── Stub Passes (Identity Passthrough) ────────────────────────────────────
    //
    // Each stub returns the input buffer unchanged.
    // Real embedding values flow through unmolested — no zeros are introduced.
    // To wire a real implementation, replace the body with a GPU dispatch.

    fn project_dimensions(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        layer_idx: usize,
        d_from: usize,
        d_to: usize,
        seq_len: usize,
    ) -> Result<wgpu::Buffer, BlockError> {
        let key = format!("projections.proj_{}.weight", layer_idx);
        let proj_w = weight_buffers
            .get(&key)
            .ok_or_else(|| block_err!(&format!("projection weight '{}' not found in VRAM", key)))?;

        let shape_data: [u32; 4] = [d_from as u32, d_to as u32, 0, 0];
        let shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("proj_shape"),
            contents: bytemuck::cast_slice(&shape_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proj_out"),
            size: (seq_len * d_to * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("proj_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/projection.wgsl").into()),
        });
        let bgl = Self::make_bgl(
            device,
            "proj_bgl",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let pipeline = Self::make_pipeline(device, "proj_pipe", &bgl, &shader, "linear_proj");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proj_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: proj_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("proj_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("proj_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((d_to * seq_len as usize + 255) / 256) as u32, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        Ok(out_buf)
    }

    fn mamba_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        layer: &AideenLayer,
    ) -> Result<wgpu::Buffer, BlockError> {
        let layer_idx = layer.idx;
        let d = layer.d_in;
        let expand = 2usize;
        let d_inner = d * expand;
        let d_state = self.config.d_state;
        let d_conv = self.config.d_conv;
        let dt_rank = (d + 15) / 16; // ceil(D / 16)

        // ── Helper: look up a mandatory weight ────────────────────────────
        macro_rules! w {
            ($key:expr) => {
                weight_buffers
                    .get(&format!("layers.{}.mamba.{}", layer_idx, $key))
                    .ok_or_else(|| {
                        block_err!(&format!(
                            "mamba layer {}: weight '{}' not found in VRAM",
                            layer_idx, $key
                        ))
                    })?
            };
        }

        let in_proj_w = w!("in_proj.weight");
        let norm_w_buf = w!("norm.weight");
        let conv1d_w = w!("conv1d.weight");
        let conv1d_b = w!("conv1d.bias");
        let x_proj_w = w!("x_proj.weight");
        let dt_proj_w = w!("dt_proj.weight");
        let dt_proj_b = w!("dt_proj.bias");
        let a_log = w!("A_log");
        let d_param = w!("D");
        let out_proj_w = w!("out_proj.weight");

        // ── Shape uniform (shared by all passes) ──────────────────────────
        let shape_data: [u32; 5] = [
            d as u32,
            d_inner as u32,
            d_state as u32,
            d_conv as u32,
            dt_rank as u32,
        ];
        let shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mamba_shape"),
            contents: bytemuck::cast_slice(&shape_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── Intermediate VRAM buffers ─────────────────────────────────────
        let mk_buf = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let proj_out_size = dt_rank + 2 * d_state;
        let xz = mk_buf("mamba_xz", 2 * d_inner);
        let x_act = mk_buf("mamba_x_act", d_inner);
        let proj_bcd = mk_buf("mamba_proj_bcd", proj_out_size);
        let delta = mk_buf("mamba_delta", d_inner);
        let y_inner = mk_buf("mamba_y_inner", d_inner);
        let output = mk_buf("mamba_output", d);

        // ── Dispatch helper ───────────────────────────────────────────────
        let dispatch = |label: &str,
                        src: &str,
                        entry: &str,
                        bgl: wgpu::BindGroupLayout,
                        bg: wgpu::BindGroup,
                        n_out: usize| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = Self::make_pipeline(device, label, &bgl, &shader, entry);
            let mut enc = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(((n_out + 255) / 256) as u32, 1, 1);
            }
            queue.submit(std::iter::once(enc.finish()));
        };

        // ─────────────────────────────────────────────────────────────────
        // Pass 1: RMSNorm + in_proj → xz[2*d_inner]
        // ─────────────────────────────────────────────────────────────────
        let bgl1 = Self::make_bgl(
            device,
            "m_bgl1",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: norm_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: in_proj_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: xz.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p1",
            include_str!("../shaders/mamba_p1_norm_proj.wgsl"),
            "rms_norm_in_proj",
            bgl1,
            bg1,
            2 * d_inner,
        );

        // ─────────────────────────────────────────────────────────────────
        // Pass 2: conv1d + SiLU → x_act[d_inner]
        // ─────────────────────────────────────────────────────────────────
        let bgl2 = Self::make_bgl(
            device,
            "m_bgl2",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: xz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: conv1d_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: conv1d_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: x_act.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p2",
            include_str!("../shaders/mamba_p2_conv.wgsl"),
            "conv1d_silu",
            bgl2,
            bg2,
            d_inner,
        );

        // ─────────────────────────────────────────────────────────────────
        // Pass 3: x_proj GEMV → proj_bcd[dt_rank + 2*d_state]
        // ─────────────────────────────────────────────────────────────────
        let bgl3 = Self::make_bgl(
            device,
            "m_bgl3",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg3"),
            layout: &bgl3,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_proj_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: proj_bcd.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p3",
            include_str!("../shaders/mamba_p3_x_proj.wgsl"),
            "x_proj",
            bgl3,
            bg3,
            proj_out_size,
        );

        // ─────────────────────────────────────────────────────────────────
        // Pass 4: dt_proj + softplus → delta[d_inner]
        // ─────────────────────────────────────────────────────────────────
        let bgl4 = Self::make_bgl(
            device,
            "m_bgl4",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg4"),
            layout: &bgl4,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: proj_bcd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dt_proj_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dt_proj_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: delta.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p4",
            include_str!("../shaders/mamba_p4_dt_proj.wgsl"),
            "dt_proj_softplus",
            bgl4,
            bg4,
            d_inner,
        );

        // ─────────────────────────────────────────────────────────────────
        // Pass 5: SSM + D skip + SiLU gate → y_inner[d_inner]
        // ─────────────────────────────────────────────────────────────────
        let bgl5 = Self::make_bgl(
            device,
            "m_bgl5",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg5 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg5"),
            layout: &bgl5,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: xz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: proj_bcd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: a_log.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: d_param.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: y_inner.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p5",
            include_str!("../shaders/mamba_p5_ssm.wgsl"),
            "ssm_step_gate",
            bgl5,
            bg5,
            d_inner,
        );

        // ─────────────────────────────────────────────────────────────────
        // Pass 6: out_proj GEMV + residual → output[D]
        // ─────────────────────────────────────────────────────────────────
        let bgl6 = Self::make_bgl(
            device,
            "m_bgl6",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let bg6 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("m_bg6"),
            layout: &bgl6,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_proj_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        dispatch(
            "m_p6",
            include_str!("../shaders/mamba_p6_out_proj.wgsl"),
            "out_proj_residual",
            bgl6,
            bg6,
            d,
        );

        Ok(output)
    }

    fn attention_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        layer: &AideenLayer,
        seq_len: usize,
    ) -> Result<wgpu::Buffer, BlockError> {
        let (light_dim, heavy_dim) = layer
            .attn_dims
            .ok_or_else(|| block_err!("attention_pass: layer has no attn_dims"))?;

        let d_model = layer.d_in;
        let layer_idx = layer.idx;

        // Create output buffer and copy input to it (residual basis)
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_out"),
            size: input.size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc_copy = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_res_copy"),
        });
        enc_copy.copy_buffer_to_buffer(&input, 0, &out_buf, 0, input.size());
        queue.submit(std::iter::once(enc_copy.finish()));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attn_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/attention.wgsl").into()),
        });

        let bgl = Self::make_bgl(
            device,
            "attn_bgl",
            &[
                wgpu::BufferBindingType::Uniform,                      // shape
                wgpu::BufferBindingType::Storage { read_only: true },  // hidden
                wgpu::BufferBindingType::Storage { read_only: true },  // norm_w
                wgpu::BufferBindingType::Storage { read_only: true },  // qkv_w
                wgpu::BufferBindingType::Storage { read_only: true },  // out_w
                wgpu::BufferBindingType::Storage { read_only: false }, // output (+acc)
            ],
        );

        let pipeline = Self::make_pipeline(device, "attn_pipe", &bgl, &shader, "attention_step");

        // Helper to run one stream (light or heavy)
        let mut run_stream =
            |name: &str, head_dim: usize, input_buf: &wgpu::Buffer| -> Result<(), BlockError> {
                let norm_w = weight_buffers
                    .get(&format!("layers.{}.attn.norm_{}.weight", layer_idx, name))
                    .ok_or_else(|| block_err!(&format!("attn.norm_{} not found", name)))?;
                let qkv_w = weight_buffers
                    .get(&format!("layers.{}.attn.qkv_{}.weight", layer_idx, name))
                    .ok_or_else(|| block_err!(&format!("attn.qkv_{} not found", name)))?;
                let out_w = weight_buffers
                    .get(&format!("layers.{}.attn.out_{}.weight", layer_idx, name))
                    .ok_or_else(|| block_err!(&format!("attn.out_{} not found", name)))?;

                let shape_data: [u32; 4] = [d_model as u32, head_dim as u32, seq_len as u32, 0];
                let shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("attn_{}_shape", name)),
                    contents: bytemuck::cast_slice(&shape_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("attn_{}_bg", name)),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shape_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: norm_w.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: qkv_w.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_w.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: out_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("attn_{}_enc", name)),
                });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("attn_{}_pass", name)),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(((d_model + 255) / 256) as u32, 1, 1);
                }
                queue.submit(std::iter::once(enc.finish()));
                Ok(())
            };

        // 1. Light stream (uses layer input)
        run_stream("light", light_dim, &input)?;

        // 2. Heavy stream (also uses layer input)
        run_stream("heavy", heavy_dim, &input)?;

        Ok(out_buf)
    }
    async fn moe_pass(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        input: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        layer: &AideenLayer,
        _seq_len: usize,
    ) -> Result<wgpu::Buffer, BlockError> {
        let d_model = layer.d_in;
        let layer_idx = layer.idx;
        let num_experts = self.config.num_total_experts;
        let expert_ffn_expand = self.config.expert_ffn_expand;
        let expert_hidden = d_model * expert_ffn_expand;

        // 1. Dispatch Router
        let logit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe_logits"),
            size: (num_experts * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let norm_w = weight_buffers
            .get(&format!("layers.{}.moe.norm.weight", layer_idx))
            .ok_or_else(|| block_err!("moe.norm not found"))?;
        let router_w = weight_buffers
            .get(&format!("layers.{}.moe.router.weight", layer_idx))
            .ok_or_else(|| block_err!("moe.router not found"))?;

        let shape_data = [d_model as u32, num_experts as u32, expert_hidden as u32, 0];
        let shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moe_shape"),
            contents: bytemuck::cast_slice(&shape_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moe_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/moe.wgsl").into()),
        });

        let bgl_router = Self::make_bgl(
            device,
            "moe_router_bgl",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let pipe_router = Self::make_pipeline(
            device,
            "moe_router_pipe",
            &bgl_router,
            &shader,
            "moe_router",
        );
        let bg_router = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moe_router_bg"),
            layout: &bgl_router,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: norm_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: router_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: logit_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_router_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("moe_router_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipe_router);
            pass.set_bind_group(0, &bg_router, &[]);
            pass.dispatch_workgroups(((num_experts + 255) / 256) as u32, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));

        // 2. Read back logits
        let logits = self.readback::<f32>(device, queue, &logit_buf).await?;

        // 3. Top-K selection (Simplified Adaptive version for single token inference)
        let mut indexed_logits: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_l = indexed_logits[0].1;
        let k = self.config.k_max;
        let sum_exp: f32 = indexed_logits
            .iter()
            .take(k)
            .map(|(_, l)| (l - max_l).exp())
            .sum();
        let top_experts: Vec<(usize, f32)> = indexed_logits
            .into_iter()
            .take(k)
            .map(|(i, l)| (i, (l - max_l).exp() / sum_exp))
            .collect();

        // 4. Dispatch Experts
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe_out"),
            size: input.size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc_res = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_res_copy"),
        });
        enc_res.copy_buffer_to_buffer(&input, 0, &out_buf, 0, input.size());
        queue.submit(std::iter::once(enc_res.finish()));

        let bgl0 = Self::make_bgl(
            device,
            "moe_exp_bgl0",
            &[
                wgpu::BufferBindingType::Uniform,                      // shape
                wgpu::BufferBindingType::Storage { read_only: true },  // raw input
                wgpu::BufferBindingType::Storage { read_only: true },  // norm_w
                wgpu::BufferBindingType::Storage { read_only: false }, // output
            ],
        );
        let bgl1 = Self::make_bgl(
            device,
            "moe_exp_bgl1",
            &[
                wgpu::BufferBindingType::Storage { read_only: true }, // fc1
                wgpu::BufferBindingType::Storage { read_only: true }, // fc2
            ],
        );
        let bgl2 = Self::make_bgl(
            device,
            "moe_exp_bgl2",
            &[
                wgpu::BufferBindingType::Uniform, // params
            ],
        );
        let layout_expert = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_exp_layout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
            push_constant_ranges: &[],
        });
        let pipe_expert = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_exp_pipe"),
            layout: Some(&layout_expert),
            module: &shader,
            entry_point: Some("expert_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        for (e_idx, weight) in top_experts {
            // Match PyTorch nn.Sequential keys: .0.weight for FC1, .2.weight for FC2
            let fc1_key = format!("layers.{}.moe.experts.{}.0.weight", layer_idx, e_idx);
            let fc2_key = format!("layers.{}.moe.experts.{}.2.weight", layer_idx, e_idx);
            let fc1_w = weight_buffers.get(&fc1_key);
            let fc2_w = weight_buffers.get(&fc2_key);

            if let (Some(w1), Some(w2)) = (fc1_w, fc2_w) {
                let params_data = [weight, 0.0];
                let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("moe_params"),
                    contents: bytemuck::cast_slice(&params_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("moe_exp_bg0"),
                    layout: &bgl0,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shape_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: norm_w.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: out_buf.as_entire_binding(),
                        },
                    ],
                });
                let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("moe_exp_bg1"),
                    layout: &bgl1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: w1.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w2.as_entire_binding(),
                        },
                    ],
                });
                let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("moe_exp_bg2"),
                    layout: &bgl2,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    }],
                });

                let mut enc_exp = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("moe_exp_enc"),
                });
                {
                    let mut pass = enc_exp.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("moe_exp_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipe_expert);
                    pass.set_bind_group(0, &bg0, &[]);
                    pass.set_bind_group(1, &bg1, &[]);
                    pass.set_bind_group(2, &bg2, &[]);
                    pass.dispatch_workgroups(((d_model + 255) / 256) as u32, 1, 1);
                }
                queue.submit(std::iter::once(enc_exp.finish()));
            }
        }

        Ok(out_buf)
    }

    // ── LM Head + Argmax (async) ─────────────────────────────────────────────
    //
    // Two GPU passes submitted in a single command buffer:
    //   Pass 1 — lm_head.wgsl : hidden[last] @ vocab.weight^T → logits[vocab_size]
    //   Pass 2 — argmax.wgsl  : parallel tree reduction → partials[num_workgroups]
    //
    // Readback: only ≤1576 bytes (≤197 partials × 8B each).
    // CPU reduces the partials to the final argmax token index.
    //
    // Async contract:
    //   slice.map_async(Read, |result| tx.send(result))
    //   device.poll(Poll)   — non-blocking, kicks GPU scheduler
    //   rx.await            — yields to JS event loop; Chrome fires callback;
    //                         tx.send() wakes the task; rx resolves
    //
    // oneshot::Sender<Result<(), BufferAsyncError>> is Send + 'static → no hacks needed.

    async fn lm_head(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: wgpu::Buffer,
        weight_buffers: &HashMap<String, wgpu::Buffer>,
        seq_len: usize,
    ) -> Result<f32, BlockError> {
        // Derive D from the actual buffer size — robust to any model variant
        let d = hidden.size() as usize / (seq_len * 4);
        let v = self.config.vocab_size;

        let norm_wt = weight_buffers
            .get("final_norm.weight")
            .ok_or_else(|| block_err!("lm_head: final_norm.weight not found"))?;
        let vocab_wt = weight_buffers
            .get("vocab_head.weight")
            .or_else(|| weight_buffers.get("lm_head.weight"))
            .or_else(|| weight_buffers.get("embedding.weight")) // tied weights fallback
            .ok_or_else(|| block_err!("lm_head: weight not found in VRAM"))?;

        // ── Pass 1: lm_head matmul ──────────────────────────────────────────
        let lm_shape: [u32; 4] = [d as u32, v as u32, 0, 0];
        let lm_shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lm_shape"),
            contents: bytemuck::cast_slice(&lm_shape),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let logit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits"),
            size: (v * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lm_head_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/lm_head.wgsl").into()),
        });
        let lm_bgl = Self::make_bgl(
            device,
            "lm_bgl",
            &[
                wgpu::BufferBindingType::Uniform,                      // shape
                wgpu::BufferBindingType::Storage { read_only: true },  // hidden
                wgpu::BufferBindingType::Storage { read_only: true },  // norm_w
                wgpu::BufferBindingType::Storage { read_only: true },  // vocab_w
                wgpu::BufferBindingType::Storage { read_only: false }, // logits
            ],
        );
        let lm_pipeline =
            Self::make_pipeline(device, "lm_pipe", &lm_bgl, &lm_shader, "lm_head_forward");
        let lm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lm_bg"),
            layout: &lm_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lm_shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hidden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: norm_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vocab_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: logit_buf.as_entire_binding(),
                },
            ],
        });

        // ── Pass 2: parallel argmax reduction ───────────────────────────────
        // Each workgroup of 256 threads reduces 256 logits → 1 partial max.
        // ≤197 workgroups for vocab=50257 → ≤1576 bytes of staging data.
        let num_wg = (v + 255) / 256;
        let partials_size = (num_wg * 8) as u64; // 8 bytes: [f32 max_val, u32 max_idx]
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax_partials"),
            size: partials_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("argmax_staging"),
            size: partials_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ax_shape: [u32; 4] = [v as u32, 0, 0, 0];
        let ax_shape_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ax_shape"),
            contents: bytemuck::cast_slice(&ax_shape),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("argmax_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/argmax.wgsl").into()),
        });
        let ax_bgl = Self::make_bgl(
            device,
            "ax_bgl",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let ax_pipeline =
            Self::make_pipeline(device, "ax_pipe", &ax_bgl, &ax_shader, "argmax_reduce");
        let ax_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ax_bg"),
            layout: &ax_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ax_shape_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: logit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partials_buf.as_entire_binding(),
                },
            ],
        });

        // ── Single command buffer: both passes + staging copy ────────────────
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lm_argmax_enc"),
        });
        {
            let mut p1 = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lm_head_pass"),
                timestamp_writes: None,
            });
            p1.set_pipeline(&lm_pipeline);
            p1.set_bind_group(0, &lm_bg, &[]);
            p1.dispatch_workgroups(((v + 63) / 64) as u32, 1, 1);
        }
        {
            let mut p2 = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("argmax_pass"),
                timestamp_writes: None,
            });
            p2.set_pipeline(&ax_pipeline);
            p2.set_bind_group(0, &ax_bg, &[]);
            p2.dispatch_workgroups(num_wg as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&partials_buf, 0, &staging_buf, 0, partials_size);
        queue.submit(std::iter::once(enc.finish()));

        // ── Async GPU readback via oneshot channel ───────────────────────────
        //
        // oneshot::Sender<Result<(), BufferAsyncError>> is Send + 'static,
        // satisfying wgpu's map_async callback requirement without any hacks.
        //
        // rx.await yields Poll::Pending to wasm_bindgen_futures' executor,
        // returning control to the JS event loop. Chrome's GPU scheduler
        // completes the work, fires the map_async callback (tx.send(result)),
        // which wakes the executor task. rx.await then returns Ok(Ok(())).
        let slice = staging_buf.slice(..);
        let (tx, rx) = oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();

        slice.map_async(wgpu::MapMode::Read, |result| {
            // tx is Send — this closure satisfies wgpu's Send + 'static requirement
            let _ = tx.send(result);
        });

        // Non-blocking poll: hints to wgpu's internal scheduler that commands are submitted.
        device.poll(wgpu::MaintainBase::Poll);

        // Await: yields to JS event loop → Chrome fires callback → task resumes
        rx.await
            .map_err(|_| block_err!("lm_head: map_async sender dropped unexpectedly"))?
            .map_err(|_| block_err!("lm_head: GPU buffer mapping failed"))?;

        // ── CPU final reduce: ≤197 partials → 1 winner ─────────────────────
        let data = slice.get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&data);
        let (mut best_val, mut best_idx) = (f32::NEG_INFINITY, 0u32);
        for wg in 0..num_wg {
            let val = f32::from_bits(raw[wg * 2]);
            let idx = raw[wg * 2 + 1];
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        drop(data);
        staging_buf.unmap();

        Ok(best_idx as f32)
    }

    // ── Pipeline Helpers ─────────────────────────────────────────────────────

    async fn readback<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Result<Vec<T>, BlockError> {
        let size = buffer.size();
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_enc"),
        });
        enc.copy_buffer_to_buffer(buffer, 0, &staging_buf, 0, size);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, |result| {
            let _ = tx.send(result);
        });

        device.poll(wgpu::MaintainBase::Poll);

        rx.await
            .map_err(|_| block_err!("readback: map_async sender dropped"))?
            .map_err(|_| block_err!("readback: GPU buffer mapping failed"))?;

        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, T>(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        Ok(result)
    }

    fn make_bgl(
        device: &wgpu::Device,
        label: &str,
        types: &[wgpu::BufferBindingType],
    ) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = types
            .iter()
            .enumerate()
            .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: *ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        })
    }

    fn make_pipeline(
        device: &wgpu::Device,
        label: &str,
        bgl: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}
