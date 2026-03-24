# AIDEEN — Decentralized AI Engine for Consumer Hardware

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
- **29 WGSL GPU compute shaders** for training and inference

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical specification.

## Current Status (March 2026)

- **GPU training pipeline:** Fully operational on AMD Radeon 780M (integrated GPU, 2 GB VRAM, Vulkan)
- **Model configuration:** d_r=512, h_slots=8, vocab_size=50,257 (BPE tokenizer)
- **Training results:** Validation loss reduced from 5.97 (random init) to 4.08 over 2,860 gradient steps on a 3.76M token corpus (Rust Book + arXiv ML papers + SmolTalk)
- **Stability:** 12+ hours continuous GPU training at 11.8 tokens/second with automatic checkpointing
- **Data ready:** 10 GB multilingual Wikipedia corpus (4.28B tokens, English + Spanish) tokenized and prepared for larger-scale training

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

```bash
# Build everything
cargo build --release --workspace

# Run tests (CPU-only crates)
cargo test --workspace --exclude aideen-block --exclude aideen-engine

# Train on a text file (requires GPU via wgpu)
cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus.txt --epochs 5

# Resume from checkpoint
cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus.txt --resume model_large --epochs 5

# Interactive chat with a trained model
cargo run --release --features wgpu -p aideen-training --bin chat -- --model model_large
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIDEEN_CTX_LEN` | 256 | Context window length (tokens) |
| `AIDEEN_BATCH_SIZE` | 1 | Sequences per gradient step |
| `AIDEEN_DEQ_HIST_GATED` | 1 | Enable history-gated mode |

## Project History

Development began in early 2025 as a private research project exploring DEQ architectures for efficient AI. The repository was migrated to GitHub in February 2026 for open-source release. Prior work included iterative prototyping of the DEQ solver, Mamba integration experiments, and the transition from Conjugate Gradient to Picard Adjoint for the backward pass. The current codebase (~15,000 lines of Rust + 5,679 lines of WGSL) represents approximately one year of cumulative R&D by two developers.

## Team

- **Juan Patricio Marchetto** ([@JuanMarchetto](https://github.com/JuanMarchetto)) — System architect, training infrastructure. Italian citizen.
- **Sergio Ariel Solis** ([@SergioAriel](https://github.com/SergioAriel)) — GPU compute, mathematical innovations (Mamba-outside-DEQ, Picard adjoint).

## Funding

AIDEEN has been self-funded to date. We are applying for [NGI Zero Commons Fund](https://nlnet.nl/commonsfund/) (EU) to support scaling experiments, published benchmarks, and a browser-based inference demo via WebGPU.

## Contributing

Contributions welcome. Please open an issue before submitting large PRs. See [ARCHITECTURE.md](ARCHITECTURE.md) for technical context.
