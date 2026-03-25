# Changelog

## [Unreleased] — March 2026

### Training
- Epoch 1 training in progress on corpus_combined (3.76M tokens), loss 5.97 → 4.65
- Validated 12+ hours continuous GPU training at 11.8 tokens/sec on AMD Radeon 780M (2GB VRAM)
- Automatic checkpointing with VRAM checksum verification before save

### Architecture
- Per-slot value projections (independent W_v per attention slot)
- Learnable forget gate on Mamba state
- Dynamic history gating with spectral norm enforcement (sigma <= 0.10)
- Renamed debug metric `hit_count` → `unconverged_count` for clarity (0 = 100% Picard convergence)

### Infrastructure
- CI expanded: check, clippy, tests, format — all green
- Benchmark suite (aideen-bench) with Candle transformer baseline, iso-data/iso-time protocol, 5-seed statistical testing

## [0.0.1] — February 2026

### Initial Open-Source Release
- 10-crate Rust workspace architecture
- DEQ fixed-point solver with Picard iteration
- Mamba-style SSM placed outside DEQ loop (convergence-safe temporal memory)
- 25 WGSL GPU compute shaders (forward, backward, update, spectral renorm)
- Picard adjoint backward pass via implicit differentiation (O(1) memory)
- CPU reference implementation with GPU validation
- BPE tokenizer (vocab 50,257) with streaming corpus processing
- P2P node architecture (QUIC/WebTransport, Ed25519 governance)
- MIT license
