# NGI Zero Commons Fund — AIDEEN Proposal Draft

**Call:** NGI Zero Commons Fund (2026-02Z)
**Deadline:** 1 April 2026, 12:00 CEST
**Requested amount:** €50,000
**Status:** DRAFT — review before submission

---

## Field 1: Proposal Name

AIDEEN — Open-Source Decentralized AI Engine for Consumer Hardware

---

## Field 2: Abstract

*"Can you explain the whole project and its expected outcome(s)?"*

AIDEEN is an open-source AI inference and training engine built entirely in Rust, designed to run large language models on consumer-grade hardware without dependence on centralized cloud providers.

Unlike transformer-based models that require expensive GPU clusters, AIDEEN uses a Deep Equilibrium Model (DEQ) architecture combined with Mamba-style selective state memory. DEQ models (Bai, Kolter & Koltun, 2019) reuse a single set of parameters across iterative refinement steps, achieving the effective depth of a deep network with only one layer's worth of parameters. This architectural property — rather than quantization or pruning — is what enables meaningful AI on laptops and commodity desktops. **Quantifying the exact parameter efficiency vs. transformers on our specific DEQ+SSM architecture is a key deliverable of this grant** (see benchmark suite in Task Breakdown). We commit to publishing results honestly, including negative findings if the architecture does not outperform equivalent transformers.

The engine is designed for peer-to-peer decentralized deployment via WebGPU/wgpu, allowing nodes to collaboratively run and train models without a central server. All communication uses a zero-trust cryptographic protocol with sealed model states verified by Ed25519 signatures.

**Expected outcomes over the grant period:**

1. A fully functional open-source training pipeline (tokenization, training, inference) validated on a 4.3-billion token multilingual corpus (English + Spanish Wikipedia).
2. A trained proof-of-concept model demonstrating DEQ convergence and text generation.
3. GPU-accelerated training and inference via wgpu/WebGPU shaders, enabling browser-based AI.
4. Published benchmarks comparing DEQ parameter efficiency vs. equivalent transformers.
5. Documentation, architecture guide, and contributor onboarding materials.

All software is written in Rust and will be released under the MIT license. The trained model weights and training data pipeline will be published as open data.

---

## Field 3: Prior Involvement

*"Have you been involved with projects or organisations relevant to this project before?"*

We are two developers who have been building AIDEEN for the past year:

- **Juan Patricio Marchetto (@JuanMarchetto)** — Rust developer, system architect, and startup founder with 107 open-source repositories on GitHub. Recently built [Noricum](https://github.com/JuanMarchetto/noricum), an open-source C-to-Rust migration CLI with LLM-powered differential testing (38,000 LOC Rust, 7 crates, 435+ tests). Serial startup founder and CTO: Co-Founder & CTO of ORO Finance, a tokenized gold protocol on Solana that raised $1.5M in pre-seed funding (468 Capital, 2025); Co-Founder & CTO of Bondum, a blockchain-based consumer loyalty platform. Previously built distributed protocol infrastructure in Rust (loxi — Distributed Vehicle Routing Protocol), developer tooling (soda — IDL code generator, 117 stars; truss — GitHub Actions validator), and decentralized applications. Member of Web3-Builders-Alliance. For AIDEEN, designed the 10-crate workspace architecture, the DEQ fixed-point solver, the streaming Rust tokenization pipeline, and the full training infrastructure. Italian citizen focused on AI sovereignty and sustainability.

- **Sergio Ariel Solis (@SergioAriel)** — Software developer and GPU compute engineer with years of professional experience as a contractor building production systems. Contributor to soda (IDL code generator, 117 stars) and loxi (distributed routing protocol in Rust). Arctic Code Vault Contributor on GitHub. For AIDEEN, designed the core mathematical innovation: the separation of Mamba selective state memory from the DEQ Picard iteration loop, solving the convergence instability that arises when stateful recurrence operates inside a fixed-point solver. Implemented the GPU backend in wgpu/WGSL including fused Picard adjoint shaders, and led the transition from Conjugate Gradient to Picard Adjoint for the backward pass.

Together we have built the complete architecture from scratch — no forks, no wrappers around existing frameworks. The codebase is approximately 15,000 lines of Rust across 10 specialized crates, plus 29 WGSL GPU compute shaders for training and inference.

**Project timeline note:** Development began in early 2025 as private research, iterating through multiple prototype architectures (initial Conjugate Gradient solver, CPU-only training, Mamba integration experiments). The GitHub repository was created in February 2026 when we migrated to open-source development for the grant application. The 47 commits in the public repository represent the stabilized codebase; the cumulative R&D spans approximately one year.

Current status: the GPU training pipeline is fully operational and actively training on consumer hardware (AMD Radeon 780M integrated GPU, 2 GB VRAM). As of March 2026, we have:

- Tokenized a 15 MB combined corpus (Rust Book + arXiv ML papers + SmolTalk) into 3.76 million BPE tokens using a custom streaming tokenizer.
- Trained a d_r=512, h_slots=8 model for 2,860 gradient steps, reducing validation loss from 5.97 (random initialization) to 4.08 — confirming that the DEQ+Mamba architecture learns effectively on GPU.
- Implemented per-slot value projections (each attention slot has its own W_v matrix), a learnable forget gate for the Mamba state, and dynamic history gating — architectural innovations that improve temporal reasoning without increasing parameter count significantly.
- Validated training stability over 12+ hours of continuous GPU execution at 11.8 tokens/second with automatic checkpointing.
- A separate 10 GB multilingual Wikipedia corpus (4.28 billion tokens, English + Spanish) is tokenized and ready for larger-scale training runs.

---

## Field 4: Requested Amount

**€50,000**

---

## Field 5: Budget Breakdown

*"Explain what the requested budget will be used for."*

| Category | Amount | Description |
|----------|--------|-------------|
| Developer time (2 developers) | €32,000 | 4 months of focused development on GPU training, benchmarks, and documentation (€4,000/month/developer) |
| Cloud GPU compute | €8,000 | Renting A100/H100 GPU instances for training larger models (d_r=1024+) and running comparative benchmarks vs. transformers |
| Hardware | €5,000 | One discrete GPU (AMD or NVIDIA) for local development and testing of wgpu shaders |
| Travel & community | €3,000 | Attending 1-2 European open-source / AI conferences to present results and build community |
| Operational costs | €2,000 | Domain, hosting, CI/CD infrastructure, documentation site |

---

## Field 6: Other Funding Sources

*"Describe any other past or present funding."*

AIDEEN has received no external funding to date. All development has been self-funded by the two founders.

We plan to apply for additional grants (CDTI NEOTEC in Spain, Smart&Start Italia) to fund longer-term development, but NGI Zero Commons Fund would be our first external support and would specifically fund the critical milestone of GPU-accelerated training and public benchmarks.

---

## Field 7: Task Breakdown with Effort Estimates

The grant deliverables are structured in three phases. Phase 1 is the **mandatory milestone** — an eliminatory gate review at 8 weeks. Phases 2 and 3 proceed only after Phase 1 is validated. P2P multi-node deployment is explicitly **out of scope** for this grant and planned as a separate follow-up effort.

### Phase 1 — Core Validation (Months 1-2) — ELIMINATORY MILESTONE

| Task | Effort | Milestone | Status |
|------|--------|-----------|--------|
| GPU training pipeline on consumer hardware | 2 person-months | 29 WGSL shaders, fused Picard adjoint, auto-checkpointing | **COMPLETED** |
| Train d_r=512 model on multilingual corpus | 1 person-month + GPU | Trained model with published weights | **IN PROGRESS** |
| Benchmark suite: DEQ vs. transformer baselines | 1.5 person-months | Published report: iso-parameter perplexity, VRAM usage, tokens/sec | Pending |
| Ablation study: DEQ vs. feedforward, with/without SSM | 0.5 person-months | Quantified contribution of each architectural component | Pending |

**Gate review criteria (week 8):** Published benchmark comparing AIDEEN DEQ+Mamba against a transformer of equivalent parameter count (via Candle), trained on the same corpus, reporting perplexity, VRAM, and throughput. If DEQ+Mamba does not demonstrate a measurable advantage in at least one dimension (parameter efficiency, VRAM, or inference speed), the team will publish an honest analysis of the results and propose a revised plan.

### Phase 2 — Accessibility & Demo (Month 3)

| Task | Effort | Milestone |
|------|--------|-----------|
| Browser inference demo via WebGPU | 1 person-month | Interactive demo at aideen.dev, zero installation |
| Documentation, architecture guide, contributor onboarding | 1 person-month | Public docs site, ARCHITECTURE.md, CONTRIBUTING.md |

### Phase 3 — Community & Release (Month 4)

| Task | Effort | Milestone |
|------|--------|-----------|
| Community sprint at FOSDEM or Rust conference | 0.5 person-months | At least 2 external contributors with merged PRs |
| Release v0.1.0 with tagged benchmarks | 0.5 person-months | Tagged release, blog post, reproducible benchmark scripts |

Total: ~8 person-months over 4 calendar months (2 developers in parallel).

**Explicitly out of scope:** P2P multi-node training and inference. The `aideen-node` and `aideen-coordinator` crates contain architectural foundations (QUIC/WebTransport, Ed25519 governance), but full integration testing is planned as a Phase 2 effort in a follow-up grant application. This grant focuses exclusively on validating the core DEQ+SSM architecture with rigorous benchmarks, a browser demo, and community foundations.

---

## Field 8: Comparison to Existing Efforts and Related Work

*"Describe existing comparable efforts and how this project differs."*

### Academic Foundations

AIDEEN builds on two lines of research:

1. **Deep Equilibrium Models (DEQ):** Bai, Kolter & Koltun (2019) showed that implicit-depth networks — where the output is the fixed point of a single repeated layer — can match the performance of deep explicit networks with O(1) memory for the backward pass via implicit differentiation. Subsequent work on stabilized DEQs (Bai et al., 2021) introduced Jacobian regularization, and Winston & Kolter (2020) proved convergence guarantees for monotone operator DEQs. Grazzi et al. (2020) analyzed iterative differentiation methods for implicit models. AIDEEN uses Picard iteration with spectral norm enforcement (σ ≤ 0.10) to guarantee contractivity, and computes gradients via the Picard adjoint equation — a practical implicit differentiation method that avoids storing intermediate iterates.

2. **Selective State Space Models (SSM):** Gu & Dao (2023) introduced Mamba, a selective SSM that achieves linear-time sequence processing by conditioning the state transition on the input. RWKV (Peng et al., 2023) pursues a similar goal via linear attention with time-dependent decay. More recently, Mamba-2 (Dao & Gu, 2024) and Griffin (De et al., 2024) explored hybrid architectures combining SSM blocks with attention.

**AIDEEN's specific contribution** is combining these two families: DEQ fixed-point iteration for depth-efficient representation learning, with Mamba-style selective state memory for temporal context. The key architectural decision — placing the SSM **outside** the Picard iteration loop — solves a convergence problem: if recurrent state participates in the fixed-point search, the Lipschitz condition L < 1 required for contraction may be violated. By treating the temporal state as frozen context during iteration and updating it post-convergence, AIDEEN preserves the convergence guarantee while retaining temporal reasoning. To our knowledge, this combination has not been explored in the published literature or in open-source implementations.

### Comparison with Open-Source Projects

| Project | Architecture | Language | GPU Backend | Training | Consumer HW | License |
|---------|-------------|----------|-------------|----------|-------------|---------|
| **AIDEEN** | DEQ + SSM | Rust | wgpu (Vulkan/Metal/DX12/WebGPU) | Yes | Yes (2GB iGPU) | MIT |
| llama.cpp | Transformer (quantized) | C++ | CUDA, Metal | No (inference only) | Yes | MIT |
| Candle | Transformer | Rust | CUDA, Metal | Yes | Partial | Apache-2.0 |
| RWKV.cpp | Linear attention (RWKV) | C++ | CUDA | No (inference only) | Yes | Apache-2.0 |
| HF Transformers | Transformer | Python | CUDA | Yes | No (cloud GPUs) | Apache-2.0 |
| Petals | Transformer (distributed) | Python | CUDA | Yes (federated) | No | MIT |

**vs. llama.cpp / GGML:** These optimize transformer inference via quantization. AIDEEN uses a fundamentally different architecture (DEQ) that is inherently parameter-efficient — the model reuses a single parameter block iteratively. Our benchmark suite (aideen-bench, using Candle for transformer baselines) will quantify this efficiency precisely.

**vs. Candle (Hugging Face Rust ML):** Candle provides Rust ML primitives and transformer implementations on CUDA/Metal. AIDEEN uses Candle as a baseline in benchmarks but targets a different architecture (DEQ+SSM) and a different GPU backend (wgpu, which includes WebGPU for browsers — not available in Candle).

**vs. RWKV / Mamba implementations:** These explore efficient alternatives to attention but use standard stacked architectures. AIDEEN combines the SSM with implicit depth (DEQ), a unique combination that trades iteration count for parameter count.

**Unique contribution:** No existing open-source project combines DEQ + SSM in a Rust-native, WebGPU-portable stack. AIDEEN is not an incremental improvement on transformers — it is an architectural alternative designed for hardware-sovereign inference.

---

## Field 9: Technical Challenges

*"What are the main technical risks and challenges?"*

1. **DEQ convergence at scale:** Our proof-of-concept at d_r=512 with 8 attention slots converges well on GPU (validation loss dropping from 5.97 to 4.08 over 2,860 steps on a 3.76M token corpus). The risk is that larger models (d_r=1024+) may require careful tuning of the spectral normalization threshold and damping factor to maintain contractivity. Mitigation: we have extensive ablation infrastructure (aideen-bench) and spectral norm enforcement built into the training loop.

2. **GPU shader correctness:** The wgpu/WGSL backend must exactly match the CPU implementation for reproducibility. Our approach: we have a CPU reference implementation with comprehensive tests, and the GPU backend is validated against it tensor-by-tensor.

3. **SSM-DEQ interaction and history gate calibration:** The selective state memory operates outside the Picard iteration loop (post-convergence update), which preserves the fixed-point contractivity guarantee. However, the history gating parameters (W_hist, gate bias) currently receive very small effective gradient steps (~1e-11) due to the conservative learning rate scaling (lr=1e-5 × grad_scale=1e-2). This means the history mechanism contributes a stable but essentially frozen context signal (hist_rms ≈ 9.5e-4, hist/inj ratio ≈ 0.25) rather than actively adapting during training. This is a known limitation: the history gate needs either a dedicated higher learning rate or a warm-up schedule to become a fully learnable component. Investigating and resolving this calibration is part of the ablation study deliverable in Phase 1.

4. **WebGPU portability:** wgpu works across Vulkan, Metal, and DX12, but WebGPU in browsers has memory limits and no shared memory between compute shaders. We may need to tile large matrix operations for browser deployment.

---

## Field 10: Ecosystem Description and Engagement Strategy

*"Describe the ecosystem around this project and how you plan to engage with it."*

**Target users:**
- Researchers exploring alternatives to transformer architectures
- Developers building privacy-preserving, offline-capable AI applications
- Organizations in regions with limited cloud access or data sovereignty requirements
- The open-source AI community (contributors to projects like llama.cpp, RWKV, etc.)

**Engagement plan with verifiable milestones:**

| Month | Action | Verifiable Outcome |
|-------|--------|--------------------|
| 1 | Publish benchmark results as reproducible scripts | Public repo with `cargo bench` instructions, CI-validated |
| 1 | Open 10+ "good first issue" tickets with clear scope | GitHub Issues tagged, linked to architecture docs |
| 2 | Architecture documentation for contributors | ARCHITECTURE.md with buffer layouts, shader guides, onboarding path |
| 2 | Outreach to Rust ML community (r/rust, Rust Zurich, Rust London) | Post with link to benchmarks and call for contributors |
| 3 | Browser inference demo live at aideen.dev | URL accessible, zero installation, text generation in-browser |
| 3 | Contact wgpu/WebGPU working group about AI compute use case | Email thread or issue on wgpu repo documenting feedback |
| 4 | Community sprint at FOSDEM, EuroRust, or NixCon | At least 2 external contributors with merged PRs |
| 4 | Release v0.1.0 with blog post and announcement | Tagged release on GitHub, blog post on project site |

**Community channels:** GitHub Discussions (primary), Matrix/Discord channel (real-time), monthly progress blog posts.

**Contributor accessibility:** The modular 11-crate architecture enables isolated contributions — a developer can work on `aideen-bench` (benchmarks) without touching the GPU shaders, or contribute documentation without understanding the Picard adjoint. Each crate has its own tests and can be developed independently.

**European dimension:**
- The lead founder (Marchetto) is an **Italian citizen** (EU national) committed to relocating to the EU to establish the project's permanent base. Relocation is planned for Q2/Q3 2026, with Italy as the primary destination. The team will incorporate as an EU-based entity (Italy) during the grant period, enabling the project to operate as European infrastructure built by European citizens. Marchetto's Italian citizenship provides full right of establishment under EU law.
- The project directly supports EU goals of digital sovereignty and reduced dependence on US/Chinese AI infrastructure, aligning with the European Strategy for Data and the EU AI Act's emphasis on transparency and local deployment
- AIDEEN enables full AI inference on-device with no cloud dependency, meaning user data never leaves the device — supporting GDPR compliance by design
- The architecture targets AMD and Intel GPUs via Vulkan/wgpu, the dominant hardware in European consumer and enterprise markets, rather than requiring NVIDIA CUDA
- The training corpus includes European languages (English, Spanish, and Italian Wikipedia), with Italian directly reflecting the lead founder's EU citizenship and cultural ties
- All outputs published under MIT license and open data, aligning with EU Open Science and Open Source Software Strategy 2020-2023

**Synergies with existing NGI Zero projects:**
- AIDEEN's decentralized inference complements projects like OpenCloud Federation (federated cloud) and TrailBase (self-hosted backend in Rust) — all part of the same vision of infrastructure that users control
- The WebGPU inference demo would benefit from browser privacy protections like those developed by JShelter (NGI Zero funded)
- AIDEEN's open model weights contribute to the same digital commons as IronCalc and other Rust-native open-source infrastructure

**Sustainability beyond the grant:**
- The core engine is self-contained with no cloud dependencies — it runs forever on local hardware
- We plan to pursue additional EU funding (NEOTEC, Smart&Start, EIC Accelerator) for continued development
- A community of contributors will maintain and extend the project
- Potential revenue model: commercial support and consulting for organizations deploying AIDEEN, while keeping the engine fully open-source
