# Stability and Damping Guide for AIDEEN

This guide explains the physical mechanisms that keep the DEQ engine stable under load and how to tune the runtime parameters.

## 1. The Picard Solver and Damping (β)

AIDEEN uses Picard iterations to find the fixed point of reasoning ($H^*$). The stability of this process is controlled through the **Damping** factor (or $\beta$-relaxation).

### Update Equation:
$$H_{next} = \beta \cdot f(H_{curr}) + (1 - \beta) \cdot H_{curr}$$

*   **High Damping (0.6 - 0.8)**: The model is extremely stable but converges more slowly. Ideal for complex logic or multi-binding tasks where the signals are noisy.
*   **Low Damping (0.1 - 0.3)**: The model is very fast but may oscillate or diverge if the weights are not perfectly normalized.

> [!TIP]
> If you see NaNs or Infs in the logs during training, the first thing you should do is lower the damping (e.g. to `0.4`) via the `AIDEEN_DAMPING=0.4` flag.

## 2. Structural Normalization

So that AIDEEN is not a "toy model", we implement mathematical protections in the data flow:

### Logit Clipping
The scores of the associative memory are clamped to `[-25.0, 25.0]` before entering the Softmax.
*   **Why**: Without this, a very strong "key" generates an infinite exponential value, which breaks the gradient and the forward pass.

### RMSNorm on Associative Context
Every signal retrieved from memory is normalized by its own magnitude (RMS).
*   **Why**: This guarantees that the "energy" injected into the DEQ is constant. The model cannot "shout" louder than the reasoning engine can process.

## 3. Runtime Flags (Modes)

| Variable | Typical Value | Description |
| :--- | :--- | :--- |
| `AIDEEN_DAMPING` | `0.4` - `0.7` | Stability of the Picard solver. |
| `AR_PAIRS_PER_SEQ` | `1` or `2` | Number of simultaneous associations (Multi-binding). |
| `AR_AUDIT` | `1` | Enables detailed memory telemetry (Entropy, MaxProb). |
| `ASSOC_ADDR_GRAD_SCALE` | `1024` | (Internal) Calibration of the memory learning rate. |

## 4. Evolution of Intelligence vs. Stability

In AIDEEN, **stability = capability**.
An unstable model cannot learn because its gradient is noise. A calibrated model (with the correct damping and normalization) can maintain coherent representations across thousands of tokens, which enables the emergence of complex behaviors such as deductive reasoning.
