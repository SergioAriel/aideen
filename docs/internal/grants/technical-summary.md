# AIDEEN: Technical Summary for Grant Reviewers

## What Is AIDEEN?

AIDEEN is a decentralized, open-source AI inference and training engine written entirely in Rust. It replaces the dominant transformer architecture with a fundamentally more efficient design: **Deep Equilibrium Models (DEQ)** combined with **Fixed-Point Memory State Space Models (SSM)**.

In a standard transformer (GPT-4, Llama, Mistral), a user query passes through 24 to 96 stacked layers, each with its own parameters. AIDEEN replaces this tower with a **single reusable computation block** that iterates to a mathematical fixed point via Picard iteration. The result is equivalent representational power with a fraction of the parameters.

AIDEEN is **hardware-agnostic by design**. Its GPU compute layer is built on `wgpu`, which compiles to Metal (Apple), Vulkan (Linux/Android), DirectX 12 (Windows), and WebGPU (browsers). This means the same model binary runs on a MacBook, a Linux server, a Windows desktop, or directly inside a web browser via WebAssembly -- with zero installation.

The system operates as a **peer-to-peer network** with zero-trust cryptographic governance. Nodes communicate over QUIC/WebTransport using a frozen, versioned binary protocol. Model updates are cryptographically signed, chain-linked, and verified before application. An independent EthicsKernel -- a non-trainable safety module that never receives gradients -- is applied to all outputs before they reach the user.

## Why It Matters

Europe faces a structural dependency on US-based AI providers (OpenAI, Anthropic, Google) for advanced AI capabilities. European organizations currently have no viable path to sovereign AI inference that does not involve renting compute from US hyperscalers or relying on US-controlled model weights.

At the same time, current large language models require expensive cloud GPUs for both training and inference, effectively excluding most of the world's population from access to advanced AI. A 70B-parameter transformer requires 35+ GB of VRAM just to load -- far beyond consumer hardware.

AIDEEN addresses both problems simultaneously:

- **Parameter efficiency**: DEQ fixed-point models achieve comparable quality with dramatically fewer parameters, enabling inference on consumer hardware (integrated GPUs, smartphones, laptops).
- **No vendor lock-in**: Fully open-source (Rust), with a frozen protocol that ensures interoperability across independent implementations.
- **Browser inference**: WebGPU support means any user with a modern browser can run AI inference locally, with zero installation and zero data leaving their device.
- **Decentralized training**: Federated expert training across heterogeneous hardware, coordinated by cryptographic governance rather than centralized cloud infrastructure.

## Technical Differentiation

### DEQ Fixed-Point Architecture

| Property | Transformer | AIDEEN DEQ |
|----------|------------|------------|
| Parameter scaling | O(N) -- one set per layer | O(1) -- one reusable block |
| Layer count | 24-96 stacked layers | 1 block, iterated to convergence |
| Training memory | Stores all activations (O(N)) | Implicit differentiation (O(1)) |
| Convergence guarantee | N/A (fixed depth) | Picard iteration with spectral normalization |

The DEQ block contains a **FixedPointMemoryReasoning** module that combines cross-slot attention, Fixed-Point Memory SSM memory, and spectral normalization. Multiple reasoning slots (K=8-16) operate in parallel, each maintaining independent state that converges to a shared fixed point.

### Implicit Differentiation

Training does not require backpropagation through all iterations. Instead, AIDEEN uses the implicit function theorem: once the forward pass converges to fixed point h*, the gradient is computed by solving a single linear system. This reduces training memory from O(iterations) to O(1).

### WebGPU Inference

All GPU compute is implemented as WGSL shaders that compile to WebGPU. This enables:

- **Browser inference**: Any modern browser (Chrome, Firefox, Safari, Edge) can run AIDEEN models directly, with no plugins, no installation, and no data sent to external servers.
- **Cross-platform parity**: The same shader code runs natively on Metal, Vulkan, DX12, and WebGPU.

### Frozen Protocol v1

The network protocol is frozen and versioned. Core constants (`VOCAB_SIZE=64,000`, `D_GLOBAL=2048`, `MEMORY_SLOTS=16`) are immutable within a protocol version. Any change requires a major version increment, ensuring all nodes on the network remain interoperable.

### EthicsKernel

A non-trainable safety module applied to all model outputs. It is loaded at runtime, never receives gradients (`dL/d0_ethics = 0`), and cannot be modified by training or configuration. This architectural guarantee prevents safety erosion through fine-tuning or adversarial training.

## Architecture

```
User Query --> Tokenizer --> Embedding --> DEQ (Picard iteration) --> LmHead --> Response
                                            ^ |
                                       FixedPointMemoryReasoning
                                       (cross-slot attention
                                        + SSM memory
                                        + spectral normalization)
```

**Workspace crates:**

| Crate | Role |
|-------|------|
| `aideen-core` | Public contracts, sealed protocol constants |
| `aideen-reasoning` | Trainable reasoning engine (DEQ + Fixed-Point Memory) |
| `aideen-block` | Fixed-Point Memory + Attention + MoE computation block |
| `aideen-engine` | GPU compute runtime (wgpu, WGSL shaders) |
| `aideen-backbone` | Model architecture, generation strategies |
| `aideen-node` | P2P network node (QUIC, WebTransport) |
| `aideen-coordinator` | Cryptographic governance, key delegation |
| `aideen-critic` | Learning plane, quality evaluation |
| `aideen-training-lab` | Training pipeline and experimentation |
| `aideen-bench` | Benchmarking: DEQ+SSM vs Transformer iso-parameter |

## Hardware Compatibility

| Tier | Hardware | Performance |
|------|----------|-------------|
| Mobile | Smartphone GPU (Adreno, Mali, Apple GPU) | Basic inference |
| Desktop | iGPU (AMD Radeon 780M, Apple M1/M2/M3) | Full inference + light training |
| Server | Discrete GPU (NVIDIA, AMD, Intel Arc) | Full training + multi-model serving |
| Browser | WebGPU (any modern browser) | Inference only, zero install |

## Expert System

AIDEEN supports 18 domain experts trained via federated learning and distilled into the backbone:

- **Knowledge domains**: math, code, logic, NLP, science, creative, legal, medical, history, finance, philosophy, multilingual
- **Meta-reasoning**: reasoning, planning
- **Cognitive infrastructure** (no manual training): memory, synthesis, critic, general

Expert training uses a 4-phase pipeline: (1) Decomposer, (2) Backbone, (3) Federated Experts, (4) Distillation. Phases 3 and 4 can run across heterogeneous hardware connected via the P2P network.

## Key Metrics

- **Language**: 100% Rust (including GPU shaders in WGSL)
- **Protocol**: v1.0, frozen, binary (bincode over QUIC)
- **Tokenizer**: BPE 64K multilingual
- **Model dimension**: D_GLOBAL=2048
- **Memory slots**: 16 per session
- **Security**: Ed25519 signatures, epoch-based anti-replay, chain-linked updates
