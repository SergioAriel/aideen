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

Unlike transformer-based models that require expensive GPU clusters, AIDEEN uses a Deep Equilibrium Model (DEQ) architecture combined with Fixed-Point Memory-style selective state memory. DEQ models reuse a single set of parameters across iterative refinement steps, achieving the effective depth of a 16-layer transformer with only one layer's worth of parameters. Based on DEQ theory (Bai et al., 2019), this architecture achieves the effective depth of a deep network with a single layer's worth of parameters, enabling meaningful AI on laptops and commodity desktops. Quantifying the exact parameter efficiency vs. transformers on our specific architecture is a key deliverable of this grant (see benchmark suite in Task Breakdown).

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

- **Juan Patricio Marchetto (@JuanMarchetto)** — Rust developer, system architect, and startup founder with 107 open-source repositories on GitHub. Serial startup founder and CTO: Co-Founder & CTO of ORO Finance, a tokenized gold protocol on Solana that raised $1.5M in pre-seed funding (468 Capital, 2025); Co-Founder & CTO of Bondum, a blockchain-based consumer loyalty platform. Previously built distributed protocol infrastructure in Rust (loxi — Distributed Vehicle Routing Protocol), developer tooling (soda — IDL code generator, 117 stars; truss — GitHub Actions validator), and decentralized applications. Member of Web3-Builders-Alliance. For AIDEEN, designed the 10-crate workspace architecture, the DEQ fixed-point solver, the streaming Rust tokenization pipeline, and the full training infrastructure. Italian-Argentine dual citizen focused on AI sovereignty and sustainability.

- **Sergio Ariel Solis (@SergioAriel)** — Software developer and GPU compute engineer with years of professional experience as a contractor building production systems. Contributor to soda (IDL code generator, 117 stars) and loxi (distributed routing protocol in Rust). Arctic Code Vault Contributor on GitHub. For AIDEEN, designed the core mathematical innovation: the separation of Fixed-Point Memory selective state memory from the DEQ Picard iteration loop, solving the convergence instability that arises when stateful recurrence operates inside a fixed-point solver. Implemented the GPU backend in wgpu/WGSL including fused Picard adjoint shaders, and led the transition from Conjugate Gradient to Picard Adjoint for the backward pass.

Together we have built the complete architecture from scratch — no forks, no wrappers around existing frameworks. The codebase is approximately 15,000 lines of Rust across 10 specialized crates, plus 29 WGSL GPU compute shaders for training and inference.

**Project timeline note:** Development began in early 2025 as private research, iterating through multiple prototype architectures (initial Conjugate Gradient solver, CPU-only training, Fixed-Point Memory integration experiments). The GitHub repository was created in February 2026 when we migrated to open-source development for the grant application. The 47 commits in the public repository represent the stabilized codebase; the cumulative R&D spans approximately one year.

Current status: the GPU training pipeline is fully operational and actively training on consumer hardware (AMD Radeon 780M integrated GPU, 2 GB VRAM). As of March 2026, we have:

- Tokenized a 15 MB combined corpus (Rust Book + arXiv ML papers + SmolTalk) into 3.76 million BPE tokens using a custom streaming tokenizer.
- Trained a d_r=512, h_slots=8 model for 2,860 gradient steps, reducing validation loss from 5.97 (random initialization) to 4.08 — confirming that the DEQ+Fixed-Point Memory architecture learns effectively on GPU.
- Implemented per-slot value projections (each attention slot has its own W_v matrix), a learnable forget gate for the Fixed-Point Memory state, and dynamic history gating — architectural innovations that improve temporal reasoning without increasing parameter count significantly.
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

| Phase | Task | Effort | Milestone |
|-------|------|--------|-----------|
| **Month 1** | GPU training pipeline: port CPU trainer to wgpu shaders | 2 person-months | **COMPLETED** — GPU training runs on consumer AMD GPU (Radeon 780M) with 29 WGSL compute shaders, fused Picard adjoint backward pass, and automatic checkpointing |
| **Month 1-2** | Train d_r=512 and d_r=1024 models on multilingual corpus (English, Spanish, Italian) | 1 person-month + GPU compute | In progress — d_r=512 model training, scaling to full corpus with cloud GPU |
| **Month 2** | Benchmark suite: compare AIDEEN DEQ vs. transformer baselines (iso-parameter, iso-compute, inference speed) | 1.5 person-months | Published benchmark report with reproducible scripts |
| **Month 2-3** | Browser inference demo via WebGPU | 1 person-month | Interactive demo running in-browser, zero installation |
| **Month 3-4** | Documentation, contributor guide, architecture walkthrough | 1 person-month | Public docs site, README, contribution guide |
| **Month 4** | Community building, conference presentation, release v0.1.0 | 0.5 person-months | Tagged release, blog post, community channels |

Total: ~7 person-months over 4 calendar months (2 developers in parallel).

**Note on P2P decentralized inference:** The P2P networking layer (QUIC/WebTransport, zero-trust protocol) is architecturally designed and partially implemented in the `aideen-node` crate. Full integration testing and multi-node deployment are planned as a Phase 2 effort, either self-funded or as a follow-up grant application. This grant focuses on validating the core DEQ+SSM architecture with published benchmarks and a browser demo.

---

## Field 8: Comparison to Existing Efforts

*"Describe existing comparable efforts and how this project differs."*

**vs. llama.cpp / GGML:** These projects optimize transformer inference for consumer hardware through quantization. AIDEEN takes a fundamentally different approach — instead of shrinking transformers, we use a different architecture (DEQ) that is inherently parameter-efficient. A DEQ model reuses a single parameter block iteratively, achieving effective depth comparable to multi-layer transformers. Our benchmark suite (aideen-bench) is designed to quantify this efficiency precisely, with iso-parameter and iso-compute comparisons as a grant deliverable.

**vs. Hugging Face Transformers:** HF provides a broad ecosystem for running and fine-tuning transformer models. AIDEEN is not a framework for existing models — it is a new architecture designed from the ground up for efficiency and decentralization. There is no Python dependency; the entire stack is Rust.

**vs. Petals / BitTorrent-style distributed inference:** Petals distributes transformer layers across nodes. AIDEEN's DEQ architecture has only one reusable layer, making distribution simpler — nodes share the same parameters and can independently verify convergence, enabling zero-trust collaboration.

**vs. RWKV / Fixed-Point Memory implementations:** These explore efficient alternatives to attention. AIDEEN combines DEQ (fixed-point iteration) with selective state memory (Fixed-Point Memory-style), a combination that has not been explored in open-source. The key innovation is placing the SSM outside the DEQ convergence loop to maintain stability. Recent additions include per-slot value projections, a learnable forget gate on the Fixed-Point Memory state, and dynamic history gating — giving each attention slot its own temporal memory profile.

**Unique contribution:** No existing open-source project combines DEQ + SSM in a Rust-native, WebGPU-portable, P2P-ready stack. AIDEEN is not an incremental improvement on transformers — it is an architectural alternative designed for a decentralized, hardware-sovereign internet.

---

## Field 9: Technical Challenges

*"What are the main technical risks and challenges?"*

1. **DEQ convergence at scale:** Our proof-of-concept at d_r=512 with 8 attention slots converges well on GPU (validation loss dropping from 5.97 to 4.08 over 2,860 steps on a 3.76M token corpus). The risk is that larger models (d_r=1024+) may require careful tuning of the spectral normalization threshold and damping factor to maintain contractivity. Mitigation: we have extensive ablation infrastructure (aideen-bench) and spectral norm enforcement built into the training loop.

2. **GPU shader correctness:** The wgpu/WGSL backend must exactly match the CPU implementation for reproducibility. Our approach: we have a CPU reference implementation with comprehensive tests, and the GPU backend is validated against it tensor-by-tensor.

3. **SSM-DEQ interaction:** As described above, the selective state memory must operate outside the Picard iteration loop. We have solved this architecturally (temporal_step runs after convergence), but the history gate calibration — how much temporal context enters the DEQ as fixed context — requires careful tuning. This is active research.

4. **WebGPU portability:** wgpu works across Vulkan, Metal, and DX12, but WebGPU in browsers has memory limits and no shared memory between compute shaders. We may need to tile large matrix operations for browser deployment.

---

## Field 10: Ecosystem Description and Engagement Strategy

*"Describe the ecosystem around this project and how you plan to engage with it."*

**Target users:**
- Researchers exploring alternatives to transformer architectures
- Developers building privacy-preserving, offline-capable AI applications
- Organizations in regions with limited cloud access or data sovereignty requirements
- The open-source AI community (contributors to projects like llama.cpp, RWKV, etc.)

**Engagement plan:**
- All code on GitHub under MIT license from day one
- Monthly progress blog posts during the grant period
- Architecture documentation designed for contributors (not just users)
- Benchmark results published as reproducible scripts
- Browser demo as a zero-friction entry point (no installation needed)
- Presentation at 1-2 conferences (e.g., FOSDEM, NixCon, or Rust-focused events)

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
