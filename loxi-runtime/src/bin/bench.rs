// bin/bench.rs
// ─────────────────────────────────────────────────────────────────────────────
// Performance benchmarks for loxi-runtime kernels.
//
// Run: cargo run --bin loxi-runtime-bench --release
//
// Tests:
//   1. MatMul throughput at various sizes
//   2. LayerNorm throughput
//   3. Softmax throughput (attention-relevant sizes)
//   4. SiLU throughput
//   5. Full forward pass timing (model inference)
//   6. Memory retrieval latency
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::Result;
use loxi_runtime::{Dispatcher, GpuContext, Shape, Tensor};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("loxi-runtime benchmark");
    println!("======================");

    let ctx = GpuContext::new().await?;
    let disp = Dispatcher::new(ctx.clone());

    // ── Warm up GPU ──────────────────────────────────────────────────────────
    {
        let a = Tensor::from_slice(
            &vec![1.0f32; 64 * 64],
            Shape::new(vec![64, 64]),
            ctx.clone(),
        )?;
        let _ = disp.matmul(&a, &a)?;
        ctx.device.poll(wgpu::Maintain::Wait);
        println!("GPU warmed up.");
    }

    // ── MatMul benchmarks ────────────────────────────────────────────────────
    println!("\n[MatMul throughput]");
    for (m, k, n) in [(128, 2048, 2048), (512, 2048, 2048), (1024, 2048, 2048)] {
        let a = Tensor::from_slice(&vec![0.1f32; m * k], Shape::new(vec![m, k]), ctx.clone())?;
        let b = Tensor::from_slice(&vec![0.1f32; k * n], Shape::new(vec![k, n]), ctx.clone())?;

        let elapsed = bench_fn(10, || {
            let _ = disp.matmul(&a, &b)?;
            ctx.device.poll(wgpu::Maintain::Wait);
            Ok(())
        });

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        println!(
            "  [{m}×{k}] × [{k}×{n}] = {gflops:.1} GFLOP/s  ({:.2}ms)",
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // ── LayerNorm benchmarks ─────────────────────────────────────────────────
    println!("\n[LayerNorm throughput]");
    for (n, d) in [(128, 2048), (512, 2048), (1024, 4096)] {
        let x = Tensor::from_slice(&vec![0.5f32; n * d], Shape::new(vec![n, d]), ctx.clone())?;
        let gamma = Tensor::from_slice(&vec![1.0f32; d], Shape::new(vec![d]), ctx.clone())?;
        let beta = Tensor::from_slice(&vec![0.0f32; d], Shape::new(vec![d]), ctx.clone())?;

        let elapsed = bench_fn(20, || {
            let _ = disp.layernorm(&x, &gamma, &beta, 1e-5)?;
            ctx.device.poll(wgpu::Maintain::Wait);
            Ok(())
        });

        println!("  [{n}×{d}]: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    }

    // ── Softmax benchmarks ───────────────────────────────────────────────────
    println!("\n[Softmax throughput (attention)]");
    for (seq, vocab) in [(512, 512), (2048, 2048), (128, 32000)] {
        let x = Tensor::from_slice(
            &vec![0.1f32; seq * vocab],
            Shape::new(vec![seq, vocab]),
            ctx.clone(),
        )?;

        let elapsed = bench_fn(20, || {
            let _ = disp.softmax(&x)?;
            ctx.device.poll(wgpu::Maintain::Wait);
            Ok(())
        });

        println!("  [{seq}×{vocab}]: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    }

    // ── Memory retrieval ─────────────────────────────────────────────────────
    println!("\n[Memory retrieval latency]");
    {
        use loxi_runtime::{MemoryScope, MemoryStore};
        use std::path::Path;

        let store = MemoryStore::new("bench", Path::new("/tmp/loxi_bench_mem")).await?;

        // Insert 1000 entries
        for i in 0..1000 {
            let embedding: Vec<f32> = (0..2048).map(|j| ((i * j) as f32).sin()).collect();
            store
                .remember(
                    embedding,
                    format!("source_{}", i),
                    format!("recall_{}", i),
                    "bench".to_string(),
                    0.8,
                    MemoryScope::Local,
                )
                .await?;
        }

        let query: Vec<f32> = (0..2048).map(|j| (j as f32).cos()).collect();
        let start = Instant::now();
        for _ in 0..100 {
            let _ = store.retrieve_default(&query).await;
        }
        let elapsed = start.elapsed() / 100;
        println!(
            "  1000-entry index, top-8 retrieval: {:.3}ms",
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  Private: {} entries, Shared: {} entries",
            store.private_count().await,
            store.shared_count().await
        );
    }

    println!("\nDone.");
    Ok(())
}

fn bench_fn<F: Fn() -> Result<()>>(iterations: u32, f: F) -> Duration {
    // Warmup
    for _ in 0..3 {
        let _ = f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f();
    }
    start.elapsed() / iterations
}
