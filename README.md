# AIDEEN — Decentralized AI Engine for Consumer Hardware

[![CI](https://github.com/SergioAriel/aideen/actions/workflows/ci.yml/badge.svg)](https://github.com/SergioAriel/aideen/actions/workflows/ci.yml)
![Rust](https://img.shields.io/badge/Rust-35k_LOC-orange)
![WGSL](https://img.shields.io/badge/WGSL-24_shaders-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Open-source AI inference and training engine built entirely in Rust. Uses **Deep Equilibrium Models (DEQ)** combined with **Mamba-style selective state memory (SSM)** instead of stacked transformer layers — achieving comparable quality with significantly fewer parameters.

Designed to run on consumer GPUs (AMD, Intel, NVIDIA) via [wgpu](https://wgpu.rs/) / WebGPU, without dependence on CUDA or cloud providers.

**License:** MIT

---

## Architecture

AIDEEN uses a single reusable parameter block refined via Picard iteration (fixed-point solving), instead of stacking 16-96 transformer layers. The Mamba SSM provides temporal memory across tokens but operates **outside** the DEQ convergence loop — a key design decision that preserves the contractivity required for stable fixed-point convergence.

Key components:
- **DEQ fixed-point solver** with Picard iteration and spectral normalization
- **Multi-slot attention** (h_slots parallel reasoning heads with per-slot Q/K/V/W_in)
- **Mamba SSM** with selective state (input-dependent decay), forget gate, and dynamic history gating
- **Picard adjoint** backward pass via implicit differentiation (O(1) memory)
- **24 WGSL GPU compute shaders** for training and inference

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical specification.

## Current Status (March 2026)

**Two development versions validated on consumer hardware:**

**v0.1 — Stable training pipeline:**
- GPU training on AMD Radeon 780M (integrated GPU, 2 GB VRAM, Vulkan)
- Model: d_r=512, h_slots=8, vocab_size=50,257 (BPE), ~213 MB checkpoint
- Validation loss: 5.96 → 4.66 ± 0.35 sustained avg (best sustained 3.99) over 7,310 chunks (25.2% of corpus)
- Throughput: 11 tokens/second, 0% unconverged Picard iterations
- 27+ hours of stable continuous GPU training with automatic checkpointing

**v0.2 — GPU-optimized pipeline (2.9× throughput):**
- 32 tokens/second (2.9× improvement over v0.1)
- Batched command encoders, Anderson acceleration, ping-pong adjoint, tiled matmul
- Per-slot W_o architecture improvement
- 7,290 chunks processed in 16 hours from scratch

**Benchmark (DEQ vs Transformer):**
- Initial run (100K tokens, 3 seeds, p=0.0001): AIDEEN 4.17 vs Transformer 2.98
- Note: initial benchmark used suboptimal DEQ hyperparameters (max_iters=6, renorm_every=50) causing 100% unconverged iterations. Corrected benchmark (max_iters=16, renorm_every=4, matching training pipeline) pending
- DEQ convergence verified in full training pipeline: 0% unconverged at 5.9 avg iterations

**Data ready:** 10 GB multilingual Wikipedia corpus (4.28B tokens) tokenized for larger-scale training

For the key architectural insight (why Mamba runs outside the DEQ loop), see [ARCHITECTURE.md](ARCHITECTURE.md).

## Workspace Structure

| Crate | Purpose |
|-------|---------|
| `aideen-core` | Public contracts, cryptographic types, sealed protocol constants |
| `aideen-backbone` | Model architecture (DEQ + Mamba composition), tokenizer, generation |
| `aideen-block` | GPU compute block (wgpu shaders for forward/backward/update) |
| `aideen-training-lab` | Training pipeline, optimizer, checkpointing |
| `aideen-engine` | GPU compute runtime |
| `aideen-node` | P2P network node (QUIC, WebTransport) |
| `aideen-coordinator` | Cryptographic governance, key delegation |
| `aideen-reasoning` | Trainable reasoning engine |
| `aideen-critic` | Quality evaluation module (non-trainable, safety) |
| `aideen-bench` | Benchmarking: DEQ+SSM vs Transformer (iso-parameter comparison) |

## Quick Start

**Prerequisites:** Rust toolchain (stable), a GPU with Vulkan/Metal/DX12 support (for training/inference). No Python, no CUDA required.

```bash
# Clone and build
git clone https://github.com/SergioAriel/aideen.git
cd aideen
cargo build --release --workspace

# Run tests (CPU-only — no GPU needed)
cargo test --workspace --exclude aideen-block --exclude aideen-engine --exclude aideen-node

# Train on a text file (requires GPU via wgpu)
cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus.txt --epochs 5

# Resume from checkpoint (--skip-chunks to skip already-trained chunks)
cargo run --release -p aideen-training --features aideen-training/wgpu --bin train -- --file corpus.txt --resume model_large --skip-chunks 7310 --epochs 1

# Interactive chat with a trained model
cargo run --release -p aideen-training --features aideen-training/wgpu --bin chat -- --model model_large

# Run DEQ vs Transformer benchmarks (CPU, no GPU needed)
cargo run --release -p aideen-bench --bin arch_bench
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIDEEN_CTX_LEN` | 256 | Context window length (tokens) |
| `AIDEEN_BATCH_SIZE` | 1 | Sequences per gradient step |
| `AIDEEN_DEQ_HIST_GATED` | 1 | Enable history-gated mode |

## Distribution

**From source** (current): Clone and `cargo build --release`. Single static binary, no runtime dependencies beyond GPU drivers.

**Planned:**
- **Browser inference:** WebGPU via `wasm32-unknown-unknown` target (Phase 2 deliverable — requires model trained first)
- **Pre-built binaries:** GitHub Releases with Linux/macOS/Windows builds (post v0.1.0)
- **Container:** Dockerfile for reproducible builds (post v0.1.0)

## Project History

Development began in early 2025 as a private research project exploring DEQ architectures for efficient AI. The repository was migrated to GitHub in February 2026 for open-source release. Prior work included iterative prototyping of the DEQ solver, Mamba integration experiments, and the transition from Conjugate Gradient to Picard Adjoint for the backward pass. The current codebase (~35,000 lines of Rust + 5,679 lines of WGSL across 10 crates) represents approximately one year of cumulative R&D by two developers.

## Team

- **Juan Patricio Marchetto** ([@JuanMarchetto](https://github.com/JuanMarchetto)) — System architect, training infrastructure. Italian citizen.
- **Sergio Ariel Solis** ([@SergioAriel](https://github.com/SergioAriel)) — GPU compute, mathematical innovations (Mamba-outside-DEQ, Picard adjoint).

## Funding

AIDEEN has been self-funded to date. We are applying for [NGI Zero Commons Fund](https://nlnet.nl/commonsfund/) (EU) to support scaling experiments, published benchmarks, and a browser-based inference demo via WebGPU.

## Contributing

Contributions welcome. Please open an issue before submitting large PRs. See [ARCHITECTURE.md](ARCHITECTURE.md) for technical context.
