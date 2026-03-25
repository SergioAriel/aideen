---
name: continuous-improvement-loop
description: Continuous improvement workflow embedded into inspections, cleanups, refactors, debugging, tuning, or feature work. Use when any code change or investigation should also surface and triage system‑level improvements without derailing the main task.
---

# Continuous Improvement Loop

## Overview

Integrate a lightweight improvement scan into any coding task. Capture opportunities, triage them, and keep the main task on track while building a reliable improvement backlog.

## Workflow

1) **Keep the primary task dominant**
- Do not change scope unless an improvement blocks correctness or performance.
- If an improvement is non‑blocking, capture it and defer.

2) **Run the improvement scan at natural checkpoints**
- After reading code, after fixing a bug, or after tests.
- Limit to 3–5 items max; prioritize clarity over volume.

3) **Classify each improvement**
- **Structural**: fixes invariants, correctness, or stability.
- **Performance**: reduces latency, memory, or GPU work.
- **Maintainability**: reduces complexity, duplication, or confusion.
- **Observability**: improves logs, metrics, tests.

4) **Triage and decide**
- If it blocks correctness: do it now.
- If it is small and low risk: do it now.
- If it is medium/high risk: log it and defer.

5) **Log improvements consistently**
- Add to a visible backlog (comment or doc).
- Include: location, reason, expected impact, and test/metric to validate.

## Decision Rules

- Do not introduce "patches" that hide symptoms without restoring invariants.
- Prefer changes that are stable across seeds, configs, and profiles.
- If a change is risky, isolate with a flag and define removal criteria.

## Output Template (inline log)

- **Item**: short name
- **Type**: structural/perf/maintainability/observability
- **Why**: invariant or bottleneck
- **Scope**: file/line or module
- **Validation**: metric/test to confirm impact
- **Decision**: do now / defer
