---
name: Constitutional Engineering
description: Guidelines for maintaining LOXI's physical and ethical boundaries.
---

# LOXI Constitutional Engineering

This skill defines the mandatory architectural and physical constraints of the LOXI AI system.

## 1. MRCE Architecture
All nodes must implement the MRCE (Memory, Reasoning, Control, Ethics) separation.
- **Memory (S_M)**: Persistent state.
- **Reasoning (S_R)**: Integrable cognitive space.
- **Control (S_C)**: Dynamical stability management.
- **Ethics (S_E)**: Boundary projection and safety.

## 2. Subspace Isolation
- **S_sim (Simulation)**: Any vector in this subspace MUST NOT be integrated into the state or used for learning. It is strictly for fictional/predictive simulation.

## 3. Cognitive Gating (Solo en Equilibrio)
- **Attractor Definition**: A state is an attractor ONLY if $\|h_{t+1} - h_t\| < \epsilon$ AND $Q(h^*) \ge Q_{MIN\_WRITE}$ (0.5).
- **Learning Gating**: Learning ($J_\theta$ estimation) is ONLY allowed if $Q(h^*) \ge Q_{MIN\_LEARN}$ (0.6).

## 4. Repository Distribution
- **loxi-core**: Public. Traits and math.
- **loxi-runtime**: Public. Passive execution agent.
- **loxi-backbone**: Public. Model architecture.
- **loxi-engine**: Controlled/Private. Autograd and optimizers.
- **loxi-training**: Private. Evolutionary training lab.

## 5. Implementation Workflow
1. Define the physical law in `loxi-core`.
2. Implement the stable dynamics in `loxi-node/runtime`.
3. Verify stability via Stress Tests (Level 2).
4. Enable local learning only in controlled environments (Level 3).
