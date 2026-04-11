# Grant Proposal — EU NGI (Next Generation Internet)

## Programme

NGI Search / NGI Zero / NGI Commons — Next Generation Internet initiative, funded by the European Commission

---

## 1. Executive Summary

> [PLACEHOLDER: One paragraph summarizing the project. Should cover: the problem (European dependency on centralized US AI infrastructure, exclusion of most users from advanced AI due to hardware requirements), the proposed solution (AIDEEN, a decentralized AI engine built in Rust using DEQ+Fixed-Point Memory architecture with P2P networking and browser-based inference), alignment with NGI goals (open-source infrastructure, decentralized technology, privacy-preserving design, internet commons), and expected outcome (sovereign, accessible AI inference that runs in any browser with zero data leaving the user's device).]

---

## 2. Problem Statement

### 2.1 The Centralization of AI Infrastructure

Advanced AI capabilities are currently concentrated in a small number of US-based cloud providers. European citizens, organizations, and governments access AI through APIs that route data through foreign infrastructure, under foreign jurisdiction, with no visibility into model behavior or data handling.

This centralization creates three structural failures:

1. **Privacy failure**: User data (queries, documents, conversations) must leave the user's device and transit through third-party servers. Users cannot verify what happens to their data after it is sent.
2. **Access failure**: Running state-of-the-art models requires expensive cloud GPUs (a 70B-parameter transformer needs 35+ GB of VRAM). This excludes the vast majority of the world's population from direct access to advanced AI.
3. **Commons failure**: The most capable AI models are proprietary, closed-source, and controlled by a handful of corporations. There is no shared, open infrastructure for AI that serves the public interest.

### 2.2 Why Existing Open-Source Models Do Not Solve This

Open-source transformer models (Llama, Mistral, Phi) release weights but not infrastructure. They still require:

- Expensive GPU hardware or cloud rentals for inference.
- Centralized serving infrastructure (no native P2P or browser deployment).
- No protocol-level interoperability between independent deployments.
- No architectural guarantees for safety (filters can be removed by fine-tuning).

The open-source AI ecosystem has open weights but closed infrastructure. AIDEEN aims to open the infrastructure itself.

---

## 3. Proposed Solution: AIDEEN

### 3.1 Overview

AIDEEN is a decentralized AI inference and training engine written entirely in Rust. It is designed from the ground up for three properties that align directly with NGI priorities:

1. **Open-source infrastructure**: All code, protocols, and model formats are open. The protocol is frozen and versioned, enabling independent implementations.
2. **Decentralized by architecture**: P2P networking over QUIC/WebTransport with zero-trust cryptographic governance. No central server required for inference.
3. **Privacy-preserving by design**: Browser-based inference via WebGPU means the model runs locally in the user's browser. No data leaves the device. No API calls. No server logs.

### 3.2 Technical Architecture

AIDEEN replaces the transformer architecture with **Deep Equilibrium Models (DEQ)** combined with **Fixed-Point Memory State Space Models (SSM)**:

```
User Query --> Tokenizer --> Embedding --> DEQ (Picard iteration) --> LmHead --> Response
                                            ^ |
                                       FixedPointMemoryReasoning
                                       (cross-slot attention
                                        + SSM memory
                                        + spectral normalization)
```

**Key properties:**

- **O(1) parameter complexity**: One reusable computation block replaces 24-96 stacked transformer layers. The block iterates to a mathematical fixed point rather than passing through a deep stack.
- **Implicit differentiation**: Training memory is O(1) instead of O(N) -- the implicit function theorem eliminates the need to store activations for all layers.
- **Spectral normalization**: Guarantees convergence of the fixed-point iteration, providing mathematical stability guarantees that transformers lack.

### 3.3 Browser-Based Inference (Zero Install, Zero Data Leakage)

AIDEEN's GPU compute is implemented as WGSL shaders compiled via `wgpu`, which targets WebGPU natively. Combined with Rust-to-WASM compilation, this enables:

- **Full inference in any modern browser** (Chrome, Firefox, Safari, Edge).
- **Zero installation**: Users visit a URL and the model runs locally.
- **Zero data leakage**: All computation happens on the user's device. No query, no document, no conversation ever leaves the browser.
- **No backend server**: The browser is the complete inference environment.

This is a fundamentally different privacy model from API-based AI. There is no server to trust, no data to protect in transit, and no logs to audit -- because no data is transmitted in the first place.

### 3.4 P2P Network and Open Protocol

AIDEEN nodes communicate over a peer-to-peer network using a frozen, versioned binary protocol:

- **Transport**: QUIC (native) and WebTransport (browser nodes).
- **Framing**: Length-prefixed bincode over unidirectional streams.
- **Security**: Ed25519 key delegation, epoch-based anti-replay, chain-linked signed model updates.
- **Zero-trust**: Nodes must receive cryptographic delegation before participating in the learning network. Discovery signals contain only hashes, never raw model state.

The protocol is **frozen at v1**: core constants (`VOCAB_SIZE=64,000`, `D_GLOBAL=2048`, `MEMORY_SLOTS=16`) are immutable within a protocol version. Any change requires a major version increment. This guarantees interoperability between independently developed nodes, enabling a true commons infrastructure.

### 3.5 EthicsKernel: Non-Negotiable Safety

The EthicsKernel is a non-trainable safety module applied to all model outputs. It:

- Never receives gradients (`dL/d0_ethics = 0`).
- Is not included in the optimizer.
- Is loaded at runtime as a separate, immutable module.
- Cannot be disabled by configuration, fine-tuning, or adversarial training.

This is an architectural invariant, not a policy choice. It ensures that any deployment of AIDEEN -- including decentralized, uncontrolled deployments -- maintains baseline safety properties.

---

## 4. State of the Art and Differentiation

| Property | Centralized LLMs (GPT-4, Claude) | Open-Weight LLMs (Llama, Mistral) | **AIDEEN** |
|----------|----------------------------------|-----------------------------------|-----------|
| Source code | Closed | Partially open | **Fully open (Rust)** |
| Model weights | Closed | Open | **Open** |
| Protocol | Proprietary API | None (HTTP wrappers) | **Open, frozen, versioned** |
| Browser inference | No | No | **Yes (WebGPU)** |
| P2P network | No | No | **Yes (QUIC + WebTransport)** |
| Privacy | Data sent to server | Data sent to server | **Local only, zero leakage** |
| Hardware requirement | Cloud GPU | Cloud or high-end GPU | **iGPU, laptop, browser** |
| Safety guarantee | Policy-based | Removable filters | **Architectural (EthicsKernel)** |
| Decentralized | No | No | **Yes, zero-trust P2P** |

### Relevant Publications

- Bai et al., "Deep Equilibrium Models" (NeurIPS 2019)
- Gu & Dao, "Fixed-Point Memory: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Bai et al., "Stabilizing Equilibrium Models by Jacobian Regularization" (ICML 2021)

---

## 5. Work Plan

### WP1: Training Pipeline and Model Validation (Months 1-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| T1.1 | Complete the 4-phase training pipeline (Decomposer, Backbone, Federated Experts, Distillation) | End-to-end training pipeline |
| T1.2 | Train DEQ+Fixed-Point Memory backbone on multilingual corpus (BPE 64K tokenizer) | Validated backbone weights |
| T1.3 | Train at least 6 domain experts (math, code, logic, NLP, science, creative) | Expert weights + quality metrics |
| T1.4 | Benchmark against standard evaluation suites (perplexity, downstream tasks) | Comparative report vs iso-parameter transformers |

**Milestone M1 (month 4)**: Trained model with competitive perplexity on standard benchmarks.

### WP2: Browser Demo and P2P Network (Months 3-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| T2.1 | Compile inference engine to WASM with WebGPU backend | Functional WASM binary |
| T2.2 | Build web-based demo interface (browser chat, zero backend) | Publicly accessible demo URL |
| T2.3 | Deploy functional P2P network: node discovery, key delegation, signed updates | Network of at least 3 interoperating nodes |
| T2.4 | Integrate WebTransport for browser-based network participation | Browser node connected to P2P network |

**Milestone M2 (month 6)**: Public browser inference demo + functional P2P network.

### WP3: Benchmarking, Publication, and Community Building (Months 5-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| T3.1 | Iso-parameter benchmark: DEQ+SSM vs Transformer (quality, latency, memory) | Reproducible benchmark data + scripts |
| T3.2 | Write technical paper for peer-reviewed venue (NeurIPS/ICML/EMNLP) | arXiv preprint |
| T3.3 | Publish repository as open-source with full documentation | Public repository |
| T3.4 | Community onboarding: contribution guide, developer documentation, issue triage | First external contributions |

**Milestone M3 (month 8)**: Paper submitted + public repository + initial contributor community.

---

## 6. Team

| Role | Profile | Dedication |
|------|---------|------------|
| Principal Investigator / Developer 1 | Software engineer with expertise in Rust, GPU computing (wgpu/WGSL), deep learning architectures, and distributed systems. Responsible for DEQ architecture, GPU engine, and P2P protocol. | 100% |
| Developer 2 | Software engineer with expertise in Rust, language model training, and web technologies (WASM/WebGPU). Responsible for training pipeline, benchmarks, and browser demo. | 100% |

Both developers have full proficiency in Rust, enabling cross-contribution across all system components.

---

## 7. Budget Estimate

| Category | Description | Cost (EUR) |
|----------|-------------|------------|
| Personnel | 2 developers x 8 months x [monthly rate] | [TO COMPLETE] |
| Compute | GPU cloud for training (phases 1-2), test hardware | [TO COMPLETE] |
| Travel | Conferences (1-2 international), NGI events, evaluator meetings | [TO COMPLETE] |
| Other direct costs | Cloud services, domain/hosting for demo, minor equipment | [TO COMPLETE] |
| Indirect costs | Overhead (per NGI programme rules) | [TO COMPLETE] |
| **Total** | | **[TO COMPLETE]** |

*Note: Compute costs are significantly reduced by DEQ parameter efficiency. Training AIDEEN requires a fraction of the GPU cost of an equivalent transformer.*

---

## 8. Expected Impact

### 8.1 Internet Commons

- A fully open, decentralized AI infrastructure that belongs to no single entity.
- A frozen, versioned protocol that enables independent implementations to interoperate -- analogous to HTTP for AI inference.
- Open-source codebase in Rust, auditable by anyone, forkable by anyone.

### 8.2 Privacy and Data Sovereignty

- Browser-based inference eliminates data transmission entirely. The user's data never leaves their device.
- No server logs, no API keys, no data retention policies to evaluate -- because there is no server.
- Full compliance with GDPR by architectural design, not by policy.

### 8.3 Decentralization and Resilience

- P2P network with no single point of failure.
- Cryptographic governance prevents unauthorized model modifications.
- Any node can join the network with a standard browser -- no specialized infrastructure required.

### 8.4 Accessibility and Inclusion

- Inference on consumer hardware (integrated GPUs, laptops, smartphones).
- Zero-install browser access reduces the barrier to entry to a URL.
- Multilingual tokenizer (BPE 64K) designed for language diversity.

### 8.5 EU Digital Sovereignty

- European-developed AI infrastructure under European jurisdiction.
- Alternative to US-controlled AI APIs for European public and private sectors.
- Aligned with EU AI Act requirements: auditable architecture, non-removable safety module.

---

## 9. Success Indicators (KPIs)

| Indicator | Metric | Target |
|-----------|--------|--------|
| Model quality | Perplexity on standard benchmarks | Competitive with iso-parameter Llama/Mistral |
| Parameter efficiency | Quality/parameter ratio vs transformer | >= 2x improvement |
| Browser inference | Functional demo in Chrome/Firefox/Safari | Yes, publicly accessible |
| Browser latency | Tokens/second in browser (WebGPU) | >= 5 tokens/s |
| Native latency | Tokens/second on iGPU (Radeon 780M or Apple M1) | >= 10 tokens/s |
| P2P network | Interoperating nodes on protocol v1 | >= 3 nodes |
| Privacy | Data transmitted to external servers during browser inference | Zero bytes |
| Publication | Paper accepted or under review at peer-reviewed venue | >= 1 arXiv preprint |
| Open-source | Public repository with external contributors | >= 5 external contributors |
| Safety | EthicsKernel active and non-removable in all configurations | 100% output coverage |

---

## 10. Alignment with NGI Priorities

| NGI Priority | AIDEEN Alignment |
|--------------|------------------|
| **Open-source infrastructure** | 100% Rust, open-source, with frozen versioned protocol enabling independent implementations |
| **Decentralized technologies** | Native P2P architecture over QUIC/WebTransport with zero-trust cryptographic governance |
| **Privacy-preserving systems** | Browser-based inference with zero data transmission; local-only computation by design |
| **Internet commons** | Open protocol, open weights, open code; infrastructure that belongs to the commons, not to a vendor |
| **Search and discovery** | Decentralized AI inference accessible via browser, enabling next-generation search and knowledge systems |
| **Trustworthy AI** | Non-trainable EthicsKernel as architectural invariant; auditable Rust codebase; reproducible benchmarks |

---

## Annexes

- Annex A: Technical Summary of AIDEEN (see `docs/grants/technical-summary.md`)
- Annex B: Protocol v1 Specification (see `docs/protocol_v1.md`)
- Annex C: Architecture Decisions (see `ARCHITECTURE_DECISIONS.md`)
