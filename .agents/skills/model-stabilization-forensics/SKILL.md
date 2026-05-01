---
name: model-stabilization-forensics
description: Causal stabilization workflow for AIDEEN model debugging. Use when working on model stability, NaNs, convergence, recall, memory behavior, token-circuit analysis, forward/backward parity, GPU shader paths, Assoc/FPM/LM-head interactions, or any task where Codex must document the complete token path and record every investigation/change with structural rationale.
---

# Model Stabilization Forensics

Use this skill to stabilize AIDEEN by treating the model as a causal circuit, not as a pile of tunable symptoms.

## Mandatory Documents

Maintain these repository documents during the work:

- `docs/vision/token_circuit_trace.txt`: the complete token lifecycle from input token to loss, backward pass, parameter/state update, and next-token/state reuse.
- `docs/vision/model_stabilization_master_plan.txt`: the live task tree, hypotheses, validations, changes, rejected paths, logs, and next actions.

Update both documents whenever the circuit understanding changes or a stabilization task/change is added, executed, validated, rejected, or deferred.

## Core Rule

Before editing code, write or update the change gate in `model_stabilization_master_plan.txt`:

1. Target failure mode: exact bug/regression/instability.
2. Structural rationale: equation/path/invariant being corrected.
3. Anti-patch check: why it fixes the origin instead of hiding the symptom.
4. Validation plan: concrete metrics, tests, logs, and success/failure thresholds.

Do not implement a change if these four points cannot be justified.

## Full-Path Analysis Requirement

Analyze every file that affects the token path under investigation. Do not stop at the file where the symptom appears.

For every variable, function, shader binding, buffer, env flag, compile constant, or state field that participates in the path:

1. Search its definition with `rg`.
2. Search all writes/mutations.
3. Search all reads/uses.
4. Identify owner/source-of-truth.
5. Identify layout, shape, stride, and lifetime.
6. Identify whether it crosses CPU/GPU, Rust/WGSL, forward/backward, or train/eval boundaries.
7. Document what role it fulfills in `token_circuit_trace.txt` or `model_stabilization_master_plan.txt`.
8. If a related file is discovered through the search, include it in the analysis before declaring a cause.

A path is not considered understood until all transitive dependencies relevant to that path have been inspected or explicitly marked out-of-scope with a reason.

## Stabilization Method

Use this order. Do not skip ahead to quality metrics before invariants are clean.

1. Freeze a minimal corridor.
   - One benchmark/profile.
   - Fixed seed/config/flags.
   - Stable log location under `logs/`.
   - Reuse existing log files unless the run structure changes.

2. Verify invariants before quality.
   - Buffer layout and stride parity.
   - Forward/backward equation parity.
   - Compile constants and env flags match across forward/backward.
   - Buffer lifecycle: clear/zero/reuse rules.
   - Finite values: no NaN/Inf.
   - Gradients nonzero where expected.
   - Parameter updates affect the actual forward path.

3. Isolate with oracle tests.
   - Oracle write: force/select known-correct writes to test whether memory content can be preserved.
   - Oracle read: force/select known-correct bank read to test addressing and output path.
   - Oracle LM/output: inject known-correct value/logit to test LM-head coupling.
   - Use oracle paths as diagnostics only unless separately justified as architecture.

4. Change one causal piece at a time.
   - Classify every piece as measurement, layout/invariant, architecture, dynamics, or diagnostics.
   - If a composed change fails, isolate the failed piece instead of reverting useful pieces blindly.

5. Promote only with evidence.
   - No NaNs/Inf.
   - Invariants hold.
   - Target metric improves.
   - Established baseline is not degraded.
   - Result survives at least the required seeds/profiles for the corridor.

## Required Reporting

For each investigation entry in `model_stabilization_master_plan.txt`, record:

- Date/time if useful.
- Command and relevant env flags.
- Files inspected.
- Variables/functions traced.
- Hypothesis.
- Evidence.
- Change made, if any.
- Validation result.
- Classification: default stable, experimental useful, rejected, or pending.
- Next action.

For each token-circuit section in `token_circuit_trace.txt`, record:

- File/function/shader entrypoint.
- Input buffers/state.
- Output buffers/state.
- Shape/layout/stride.
- Invariants.
- Failure modes already seen.
- Diagnostics/logs that observe it.

## Stop Conditions

Stop and ask the user before proceeding if:

- A discovered dependency contradicts the current hypothesis.
- The worktree has unexpected unrelated changes in files being edited.
- The next change cannot pass the four-point change gate.
- A diagnostic requires changing default behavior without an explicit experimental flag.
## Documentation Discipline

Use documentation as an execution aid, not as paperwork.

1. Before a code change, add the four-point change gate in compact form.
2. During investigation, document only facts that change the causal picture: traced files, invariants, contradictions, commands, metrics, and decisions.
3. After validation, record the result and the next decision immediately.
4. Expand `token_circuit_trace.txt` by the path being touched, not by writing a complete encyclopedia before acting.
5. Prefer concise entries that are searchable and actionable.

If documentation starts delaying a necessary validation, write a short placeholder with the exact pending trace and continue the validation.

## Revert And Preservation Policy

Do not use broad reverts as a cleanup method during stabilization.

Classify each changed piece before reverting:

- `default stable`: validated invariant or measurement fix that should remain.
- `experimental useful`: causally plausible but not validated as default; keep behind explicit flag or document as pending.
- `diagnostic temporary`: useful observer/log/probe; keep only with removal criteria.
- `rejected`: refuted dynamics/architecture or noisy artifact; revert or disable.
- `pending`: insufficient evidence; do not promote to default.

When a composite change fails:

1. List all pieces in the package.
2. Separate measurement, layout/invariant, architecture, dynamics, and diagnostics.
3. Keep measurement/layout fixes if they restore required invariants and do not alter inactive behavior.
4. Gate or revert architecture/dynamics pieces that fail direct validation.
5. Keep temporary diagnostics only if they are still needed and have removal criteria.
6. Record the decision in `model_stabilization_master_plan.txt`.

Never revert user changes or unrelated dirty files. If an unexpected change appears in a file being edited, stop and ask the user how to proceed.

