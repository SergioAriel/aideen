// tensor.rs
// ─────────────────────────────────────────────────────────────────────────────
// Tensor: the fundamental data structure of loxi-runtime.
//
// Design principles:
//   1. GPU-first: data lives on the GPU by default.
//   2. Lazy CPU reads: pulling data to CPU is explicit (read_to_cpu).
//   3. Shape tracking: shapes are tracked in Rust, not the GPU.
//   4. Zero-copy: where possible, use views instead of copies.
//   5. Type-safe: shapes are validated at operation time.
//
// A Tensor is:
//   - A wgpu::Buffer on the GPU (the actual data)
//   - A Shape (Vec<usize>) describing its dimensions
//   - A reference to the GPU Device (for dispatching kernels)
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device, Queue};

// ─── Shape ────────────────────────────────────────────────────────────────────

/// The dimensions of a tensor.
/// Stored as Vec<usize> from outermost to innermost dimension.
/// Examples:
///   Scalar: []
///   Vector: [1024]
///   Matrix: [512, 1024]
///   4D:     [batch, heads, seq, d_head]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    /// Total number of elements (product of all dims).
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Total byte size assuming f32 elements.
    pub fn byte_size(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get a specific dimension.
    pub fn dim(&self, d: usize) -> Result<usize> {
        self.0
            .get(d)
            .copied()
            .ok_or_else(|| anyhow!("dimension {} out of range for shape {:?}", d, self.0))
    }

    /// Validate that two tensors have matching shapes for elementwise ops.
    pub fn check_broadcast(&self, other: &Shape) -> Result<()> {
        if self != other {
            return Err(anyhow!("shape mismatch: {:?} vs {:?}", self.0, other.0));
        }
        Ok(())
    }

    /// Validate matmul compatibility: self=[M,K], other=[K,N] → [M,N]
    pub fn check_matmul(&self, other: &Shape) -> Result<Shape> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(anyhow!(
                "matmul requires 2D tensors, got {:?} × {:?}",
                self.0,
                other.0
            ));
        }
        let (m, k1) = (self.0[0], self.0[1]);
        let (k2, n) = (other.0[0], other.0[1]);
        if k1 != k2 {
            return Err(anyhow!("matmul inner dims mismatch: K={} vs K={}", k1, k2));
        }
        Ok(Shape::new(vec![m, n]))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

// ─── Gpu Context ─────────────────────────────────────────────────────────────

/// Shared GPU device + queue, wrapped in Arc for cheap cloning.
/// Create one GpuContext per process and pass it everywhere.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter: Arc<wgpu::Adapter>,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| {
                anyhow!("No GPU adapter found. Ensure Metal/Vulkan/DX12 drivers are installed.")
            })?;

        let info = adapter.get_info();
        tracing::info!(
            "GPU adapter: {} ({:?}) — backend: {:?}",
            info.name,
            info.device_type,
            info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("loxi-runtime"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow!("Failed to create GPU device: {}", e))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
        })
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Create a GPU buffer from CPU data (f32 slice).
    pub fn buffer_from_slice(&self, data: &[f32], label: &str) -> Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            })
    }

    /// Create an uninitialized GPU buffer of given byte size.
    pub fn buffer_empty(&self, byte_size: u64, label: &str) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

// ─── Tensor ───────────────────────────────────────────────────────────────────

/// A multi-dimensional array living on the GPU.
///
/// Data is always f32 (single precision).
/// Shapes and operations are validated at runtime in debug builds.
///
/// Clone is cheap — it clones the Arc, not the GPU buffer.
/// To get a real copy of the data, use `.copy()`.
#[derive(Clone)]
pub struct Tensor {
    /// Shape of this tensor
    pub shape: Shape,
    /// The actual data buffer on the GPU
    pub buf: Arc<Buffer>,
    /// GPU context (device + queue)
    pub ctx: GpuContext,
    /// Optional debug name for tracing
    pub name: Option<String>,
}

impl Tensor {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a tensor from a CPU f32 slice.
    /// Data is immediately uploaded to GPU.
    pub fn from_slice(data: &[f32], shape: Shape, ctx: GpuContext) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(anyhow!(
                "data length {} doesn't match shape {} ({})",
                data.len(),
                shape,
                shape.numel()
            ));
        }
        let buf = ctx.buffer_from_slice(data, "tensor");
        Ok(Self {
            shape,
            buf: Arc::new(buf),
            ctx,
            name: None,
        })
    }

    /// Create a zero-initialized tensor of given shape.
    pub fn zeros(shape: Shape, ctx: GpuContext) -> Self {
        let data = vec![0.0f32; shape.numel()];
        let buf = ctx.buffer_from_slice(&data, "zeros");
        Self {
            shape,
            buf: Arc::new(buf),
            ctx,
            name: None,
        }
    }

    /// Create an uninitialized output tensor (allocated but garbage data).
    /// Used for operation outputs — always written before read.
    pub fn uninit(shape: Shape, ctx: GpuContext) -> Self {
        let byte_size = shape.byte_size() as u64;
        let buf = ctx.buffer_empty(byte_size, "uninit");
        Self {
            shape,
            buf: Arc::new(buf),
            ctx,
            name: None,
        }
    }

    /// Attach a debug name (fluent).
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    // ── CPU ↔ GPU Transfer ────────────────────────────────────────────────────

    /// Read tensor data back to CPU. Async — requires submitting a command
    /// and waiting for GPU to finish.
    ///
    /// WARNING: Expensive — involves GPU→CPU transfer.
    /// Only use for debugging or final output decoding.
    pub async fn read_to_cpu(&self) -> Result<Vec<f32>> {
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: self.shape.byte_size() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(&self.buf, 0, &staging, 0, self.shape.byte_size() as u64);
        self.ctx.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.await??;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        Ok(result)
    }

    /// Upload new data to this tensor from CPU.
    pub fn write_from_cpu(&self, data: &[f32]) -> Result<()> {
        if data.len() != self.shape.numel() {
            return Err(anyhow!("write_from_cpu: data length mismatch"));
        }
        self.ctx
            .queue
            .write_buffer(&self.buf, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    // ── Shape Utilities ───────────────────────────────────────────────────────

    /// Reshape tensor — returns a new Tensor that shares the same buffer.
    /// New shape must have the same numel.
    pub fn reshape(&self, new_shape: Shape) -> Result<Tensor> {
        if new_shape.numel() != self.shape.numel() {
            return Err(anyhow!(
                "reshape: cannot reshape {} to {}",
                self.shape,
                new_shape
            ));
        }
        Ok(Tensor {
            shape: new_shape,
            buf: Arc::clone(&self.buf),
            ctx: self.ctx.clone(),
            name: self.name.clone(),
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    pub fn byte_size(&self) -> usize {
        self.shape.byte_size()
    }
}

// Safe to send across threads because wgpu buffers are Send.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{} {:?}", self.shape, self.name)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        assert_eq!(Shape::new(vec![2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(vec![1024]).numel(), 1024);
    }

    #[test]
    fn test_shape_matmul_valid() {
        let a = Shape::new(vec![64, 128]);
        let b = Shape::new(vec![128, 256]);
        let c = a.check_matmul(&b).unwrap();
        assert_eq!(c, Shape::new(vec![64, 256]));
    }

    #[test]
    fn test_shape_matmul_invalid() {
        let a = Shape::new(vec![64, 128]);
        let b = Shape::new(vec![256, 512]);
        assert!(a.check_matmul(&b).is_err());
    }
}
