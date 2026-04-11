---
name: analyze-before-answer
description: Use when a question can trigger premature or contradictory answers. Forces a short evidence-first analysis before replying, separating verified facts from inference.
---

# Analyze Before Answer

Use this skill when a question is easy to answer too quickly but the cost of being wrong is high.
This includes code-state questions, architecture questions, design tradeoffs, runtime behavior, flags, equations, debugging, and any situation where answering from memory can create contradictions.

## Goal

Do not answer from momentum.
Pause, inspect the relevant evidence, then answer.

## Core rule

Before replying, identify which parts of the answer are:
- verified from files, code, logs, or measurements
- inferred from those facts
- still unknown

Do not blur those categories.

## Minimum workflow

1. Re-open the relevant evidence.
2. Check whether the question is about:
- current code state
- measured behavior
- intended design
- hypothesis / future design
3. Write the answer so the reader can tell which of those categories you are using.
4. If evidence is missing, say that directly instead of filling the gap from memory.

## When the question is about code

Do all that apply:
- inspect the file or diff
- verify whether the flag/config/path is actually used
- restate the active equation or update rule in short form if behavior depends on it
- prefer current runtime evidence over stale reasoning

## When the question is about experiments

Do all that apply:
- distinguish between one run and a robust pattern
- say which seed/recipe/profile the result came from
- do not generalize beyond the measured setup without marking it as inference

## Conflict rule

If prior statements and current evidence disagree:
- trust the current evidence
- acknowledge the mismatch plainly
- restate the corrected answer

## Response style

- start with the direct answer
- keep it short unless detail is needed
- cite files when the answer depends on current code state
- mark inference explicitly
- avoid answering both "yes" and "no" unless you first separate two different interpretations

## Minimal response scaffold

Use this only when helpful:

- `Confirmed:` ...
- `Measured:` ...
- `Inference:` ...
- `Unknown:` ...

## Repo-local reminders

In AIDEEN-like architecture discussions, this usually means checking:
- current shader/runtime path
- whether a toggle is actually consumed
- whether a component is active in baseline or only preserved as reference
- whether a claim comes from files, measurements, or an untested architectural idea
