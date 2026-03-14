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