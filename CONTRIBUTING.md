# Contributing to AIDEEN

Thank you for your interest in contributing to AIDEEN.

## Getting Started

```bash
# Clone and build
git clone https://github.com/SergioAriel/aideen.git
cd aideen
cargo build --workspace

# Run tests (CPU-only crates — GPU crates require wgpu-compatible hardware)
cargo test --workspace --exclude aideen-block --exclude aideen-engine --exclude aideen-backbone

# Run GPU tests (requires Vulkan/Metal/DX12 GPU)
cargo test --release --features wgpu -p aideen-backbone
```

## Before Submitting a PR

1. Open an issue first to discuss significant changes.
2. Ensure `cargo fmt --all` passes.
3. Ensure `cargo clippy --workspace` has no warnings.
4. Add tests for new functionality.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete technical specification of the DEQ+Mamba architecture, buffer layouts, and backward pass derivation.

## Code Organization

- `aideen-core/` — Shared types and contracts
- `aideen-backbone/` — Model architecture and GPU backend
- `aideen-block/` — wgpu compute shaders (forward, backward, update)
- `aideen-training-lab/` — Training pipeline and optimizer
- `aideen-bench/` — Benchmark suite (DEQ vs Transformer comparison)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
