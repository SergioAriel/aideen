---
description: How to implement and verify Local Learning (Nivel 3)
---

# Workflow: Local Learning Implementation

Follow these steps to safely implement or modify the local learning mechanism in a AIDEEN node.

1. **Verify Equilibrium**: Ensure the system reaches an attractor using `node.is_attractor_state(delta_norm, quality)`.
2. **Estimate Jacobian**: Call `estimate_jacobian` using finite perturbations ($\varepsilon = 1e-4$).
3. **Calculate Quality Signal**:
   - If $Q > 0.7$, assign positive reinforcement ($\eta$).
   - If $0.5 \le Q \le 0.7$, assign neutral or small update.
   - If $Q < 0.5$, ABORT.
4. **Apply Update**: Use `reasoning.apply_update(jacobian, step)` to mutate weights.
5. **Verify No-Divergence**: Run a post-update stress test to ensure the mutation hasn't broken stability.
