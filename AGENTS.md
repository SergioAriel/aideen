# Global Engineering Policy

## Root-Cause Change Gate (Mandatory Before Editing)

For every proposed code change, the agent must document and validate:

1. Target failure mode:
what exact bug/regression/instability is being addressed.

2. Structural rationale:
why the change fixes the cause at equation/path/invariant level.

3. Anti-patch check:
why the change fixes the cause instead of only hiding the symptom.
Use the following criterion:
- A change is **not a patch** if it corrects the failure at the level where it originates, restores a required invariant/property, does not rely on fragile compensations, and remains valid across seeds/profiles/bench settings.
- A change **is a patch** if it mainly suppresses symptoms, depends on narrow tuning or guardrails, breaks generality, or only keeps the system alive while the underlying mechanism remains wrong.
- Anti-patch justification must be causal. It is not enough to argue from the *type* of change made or avoided; the report must explain which mechanism/invariant was corrected and why that resolves the failure at its origin.

4. Validation plan:
which concrete metrics/tests/logs must improve to consider the change correct.

If these four points cannot be justified, do not implement the change.

## Composite Change Failure Handling

When a composed change fails validation, do not automatically revert the whole
package. First decompose it into causal pieces and classify each one.

Required process:

1. Identify the change package:
list every piece that was introduced together.

2. Separate by role:
- **Measurement**: changes that only correct how behavior is evaluated.
- **Layout/invariant**: changes that fix buffer sizes, strides, ownership, or
  other structural consistency requirements.
- **Architecture**: changes that alter capacity, topology, memory structure, or
  information flow.
- **Dynamics**: changes that alter gates, scales, softmax sharpness, read/write
  equations, optimizer behavior, or convergence behavior.
- **Diagnostics**: temporary logs, counters, and probes.

3. Validate piece by piece:
- A measurement fix may remain if it compiles and does not alter model behavior.
- A layout/invariant fix may remain if it restores a required invariant and does
  not affect inactive paths, or must be gated to the experimental path that
  requires it.
- Architecture and dynamics changes must pass direct validation before becoming
  default behavior.
- Diagnostics must remain explicitly temporary, with removal criteria.

4. Classify each piece:
- **Default stable**: validated and does not degrade the established baseline.
- **Experimental useful**: causally plausible or required for a future variant,
  but not validated as default.
- **Rejected**: hypothesis refuted, or regression introduced without a causal
  explanation and fix.
- **Pending**: needs isolated validation.

5. Revert only the regression source:
do not discard useful measurement, invariant, or diagnostic pieces just because a
different piece in the same package failed. Revert the whole package only when
the pieces are inseparable without risking semantic mismatch.

6. Report the result:
state which piece failed, which piece remains, which piece is experimental, and
the next validation required.

Operational rule: "Do not leave a failing change as default" does not mean
"delete everything that was tried." It means isolate the regression, preserve
causally useful pieces, and keep unvalidated architecture/dynamics behind an
experimental path or out of default behavior.

## Patch Definition

Do not classify a change as "real" based on the category of intervention applied or avoided.
The deciding factor is causal correctness:

- Any intervention — hyperparameters, equations, guards, layout, memory, algorithms — can be the correct locus if it addresses the failure at its origin.
- Any of those same interventions can be a patch if they only move the symptom around without restoring the broken invariant.
- Therefore, reports must not use the category of change itself as proof that a fix is structural.

## Temporary Guards

Temporary guards are allowed only if all are true:

- Explicitly labeled as temporary.
- Root-cause investigation runs in parallel.
- Clear removal criteria are defined.
