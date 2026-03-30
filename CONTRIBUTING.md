# Contributing to AIDEEN

Thank you for your interest in contributing to AIDEEN.

## Getting Started

```bash
# Clone and build
git clone https://github.com/SergioAriel/aideen.git
cd aideen
cargo build --release --workspace

# Run tests (CPU-only crates — GPU crates require wgpu-compatible hardware)
cargo test --workspace --exclude aideen-block --exclude aideen-engine --exclude aideen-backbone

# Run GPU tests (requires Vulkan/Metal/DX12 GPU)
cargo test --release --features wgpu -p aideen-backbone
```

## Running Training

```bash
# Requires a wgpu-compatible GPU (Vulkan/Metal/DX12)
cargo run --release --features wgpu -p aideen-training --bin train -- --file path/to/corpus.txt

# Resume from checkpoint
cargo run --release --features wgpu -p aideen-training --bin train -- --file corpus.txt --resume model_large
```

See the doc comment in `aideen-training-lab/src/bin/train.rs` for all CLI flags.

## How to Contribute

1. Fork the repository and create a feature branch (`git checkout -b my-feature`).
2. Open an issue first to discuss significant changes.
3. Make your changes, following the code style below.
4. Push your branch and open a Pull Request against `main`.

Look for issues labeled [**good first issue**](https://github.com/SergioAriel/aideen/labels/good%20first%20issue) if you want a place to start.

## Code Style

- Format all code with `cargo fmt --all`.
- Ensure `cargo clippy --workspace` produces no warnings.
- Add tests for new functionality.

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
