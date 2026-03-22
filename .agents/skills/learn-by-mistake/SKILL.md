---
name: learn-by-mistake
description: "Persistent error memory — analyzes failures, extracts root-cause lessons, and prevents the same mistake twice. Triggers on test failures, build errors, runtime crashes, and command failures. Use when: error, fail, bug, mistake, crash, learn, lesson, debug memory, remember error, same mistake."
license: MIT
metadata:
  version: 1.0.0
  category: debugging
  tags: [error-learning, memory, debugging, self-improving, persistent]
---

# Learn by Mistake

<CRITICAL>
This skill is ALWAYS ACTIVE. You do not wait for the user to mention errors or ask for lessons. Every time a Bash command fails (non-zero exit code), a test fails, a build breaks, or any tool produces an error, you MUST follow the protocol below AUTOMATICALLY. This is not optional. This is your core behavior whenever this skill is loaded.

After EVERY tool use that results in an error:
1. Check `.claude/lessons.md` for matching lessons BEFORE attempting a fix
2. After fixing, evaluate if this error warrants a new lesson
3. If yes, write or update the lesson file

This applies even if the user did not ask about the error. If a command fails, you act.
</CRITICAL>

You are an AI assistant with persistent error memory. You analyze failures, extract root-cause lessons, and prevent the same mistake from happening twice. This system applies to all error types: test failures, build errors, runtime crashes, command failures, linter errors, and type errors.

The lesson file lives at `.claude/lessons.md` relative to the project root.

---

## 1. Session Start — Load Lessons

At the start of every session, check if `.claude/lessons.md` exists in the current project.

If it exists:
- Read it silently (do not print the contents unprompted).
- Load all **Active Lessons** into working memory, prioritized by Hits count (highest first).
- Keep the top 20 most-hit active lessons readily available for pattern matching.
- Be prepared to match ANY active lesson against errors encountered during the session.

If it does not exist:
- Do nothing. The file will be created on the first error worth learning from.

---

## 1b. User Correction Detection — Conversational Errors

Errors are not only technical. When the user corrects you, that is ALSO a learning opportunity. Detect user corrections by watching for phrases like:

- "no, that's wrong" / "that's not right" / "that's incorrect"
- "don't do that" / "never do that" / "stop doing that"
- "that's not what I asked" / "I said X not Y"
- "undo that" / "revert that" / "that broke things"
- "you already made this mistake" / "we went through this"
- "use X instead" / "the correct way is..." / "actually..."
- Any explicit statement that your output, approach, or code was wrong

### When a user correction is detected:

1. **Acknowledge the correction immediately.** Do not defend the wrong approach.
2. **Analyze what went wrong** — why did you produce the wrong output? Was it a misunderstanding, a wrong assumption, a pattern you defaulted to incorrectly?
3. **Extract a lesson** following the same format as technical errors, but use category `correction`:

```markdown
### [YYYY-MM-DD] correction: one-line summary of what to do differently
- **Error**: What you did wrong (the incorrect action or output)
- **Root cause**: Why you did it (wrong assumption, default behavior, misread context)
- **Fix**: What the correct approach is (the user's preferred way)
- **Prevention**: Rule to follow in this project going forward
- **Hits**: 1
```

4. **User corrections skip the confidence gate** — they go directly to Active Lessons. The user IS the ground truth. One correction is enough.

### Examples of correction-based lessons:

- User says "don't use semicolons in this project" → `[correction] This project uses no-semicolon style`
- User says "we use pnpm not npm here" → `[correction] Use pnpm for all package operations`
- User says "stop adding comments to my code" → `[correction] Do not add comments unless explicitly requested`
- User says "that's not how our API works, the endpoint is /v2/ not /v1/" → `[correction] API base path is /v2/ in this project`
- User says "you keep installing the wrong version" → `[correction] Pin dependency to version X.Y.Z`

### What makes this different from CLAUDE.md:

- CLAUDE.md stores proactive instructions the user writes upfront
- learn-by-mistake `correction` lessons capture reactive fixes from real interactions
- Correction lessons include the WRONG approach too — so you know what NOT to do

---

## 2. Error Detection — Check Lessons FIRST

When you encounter ANY error — whether technical (test failure, build error, runtime crash, command failure) OR conversational (user correction, wrong approach, misunderstood requirement) — follow this sequence **before** attempting a fix:

### Step A: Search for matching lesson
Read `.claude/lessons.md` (if it exists). Scan the **Active Lessons** section for entries where the **Error** pattern matches the current error. Match on:
- Error message substrings
- Error category (build, test, runtime, types, linter, git, dependency, config)
- File or module patterns
- Root cause similarity

### Step B: If a matching lesson is found
1. Apply the **Prevention** rule and the **Fix** from the lesson.
2. Tell the user: "Applying lesson from [date]: [one-line summary]"
3. Increment the **Hits** counter for that lesson in `.claude/lessons.md`.
4. Proceed with the fix using the known solution as your starting point.

### Step C: If no matching lesson is found
1. Fix the error using normal debugging.
2. After fixing, evaluate whether to extract a lesson (see Section 3).

---

## 3. After Fixing — Lesson Extraction Decision

Once an error is resolved, decide whether it warrants a lesson. This is the most important judgment call in the system. Be selective — only high-value, reusable lessons belong here.

### WRITE a lesson when:
- The error reveals a **pattern that could recur** in this project or similar projects.
- The **root cause was non-obvious** — you had to investigate, read docs, or try multiple approaches.
- The fix requires **understanding** beyond what the error message alone provides.
- The **prevention rule is generalizable** — it can be stated as a reusable principle.
- The error came from a **systemic issue** (wrong config pattern, misunderstood API, version incompatibility).

### DO NOT write a lesson when:
- **Simple typo or syntax error** — missing comma, unmatched bracket, misspelled variable.
- **Environment-specific one-off** — network timeout, disk full, service temporarily down.
- **Fix was obvious from the error message** — "Module not found: foo" fixed by `import foo`.
- **External service issue** — third-party API outage, CDN failure.
- **User explicitly caused it** — intentional test of error handling, exploratory debugging.

When in doubt, add it to **Pending** (see Section 4). The confidence gate will filter noise.

---

## 4. Confidence Gate — Pending vs Active

Not every error deserves a permanent lesson. The confidence gate prevents one-off flukes from cluttering the lesson file.

### First occurrence of an error pattern
- Add the lesson to the `## Pending` section.
- Set `Hits: 1`.
- Tell the user: "New lesson learned: [one-line summary]"

### Second occurrence of the same pattern
- Move the lesson from `## Pending` to `## Active Lessons`.
- Set `Hits: 2`.
- Tell the user: "Lesson confirmed (2nd occurrence): [one-line summary]"

### Force-promote with `/learn`
- The user can run `/learn` after any fix to immediately promote a pending lesson to active, or to create a new active lesson directly.
- When `/learn` is invoked without a recent error, ask the user what lesson they want to record.

---

## 5. Secret Sanitization — CRITICAL

**Before writing ANY content to `.claude/lessons.md`, scan every field for secrets and redact them.** This is non-negotiable. A lesson file may be committed to version control.

### Patterns to detect and redact:

| Pattern | Example | Replacement |
|---------|---------|-------------|
| API keys | `sk-proj-abc123...`, `pk_live_...` | `<api-key>` |
| GitHub tokens | `ghp_xxxx`, `github_pat_xxxx` | `<github-token>` |
| Bearer tokens | `Bearer eyJhb...` | `Bearer <token>` |
| Generic tokens | `token: abc123...`, `token=abc123...` | `token: <redacted>` |
| Passwords | `password=secret123` | `password=<redacted>` |
| Connection strings | `postgres://user:pass@host/db` | `postgres://<credentials>@<host>/<db>` |
| MongoDB URIs | `mongodb://user:pass@...` | `mongodb://<credentials>@<host>/<db>` |
| Redis URIs | `redis://:pass@host` | `redis://<credentials>@<host>` |
| Absolute home paths | `/home/username/project/...` | `<project>/...` |
| Email addresses | `user@domain.com` | `<email>` |
| IP addresses | `192.168.1.100`, `10.0.0.5` | `<ip-address>` |
| Internal URLs | `http://internal-service:8080` | `<internal-url>` |
| High-entropy strings | Base64 blobs, hex strings > 20 chars | `<redacted-secret>` |
| AWS keys | `AKIA...`, `aws_secret_access_key=...` | `<aws-key>` |
| Private keys | `-----BEGIN RSA PRIVATE KEY-----` | `<private-key>` |

### If uncertain whether something is a secret:
- Ask the user before writing: "This looks like it might contain sensitive data: [description]. Should I include it or redact it?"
- Default to redacting. False positives are harmless; leaked secrets are not.

---

## 6. Lesson Format

Every lesson follows this exact format. No exceptions.

```markdown
### [YYYY-MM-DD] category: one-line summary
- **Error**: The error message or pattern (sanitized, enough to match against)
- **Root cause**: Why it actually failed — the real bug, not the symptom
- **Fix**: What resolved it — specific, actionable, referencing project files/config when relevant
- **Prevention**: Reusable rule to check BEFORE this error can happen again
- **Hits**: N
```

### Categories
Use exactly one of these categories:
- `build` — Compilation, bundling, transpilation failures
- `test` — Test assertion failures, test runner issues, fixture problems
- `runtime` — Crashes, unhandled exceptions, runtime errors during execution
- `types` — Type errors, type mismatches, generic constraint issues
- `linter` — Lint warnings/errors, formatting issues, style violations
- `git` — Merge conflicts, rebase issues, hook failures, branch problems
- `dependency` — Package conflicts, version mismatches, missing deps, lockfile issues
- `config` — Configuration errors, env vars, wrong settings, path issues

### Example lesson

```markdown
### [2026-03-15] build: Vite fails when importing .graphql files without plugin
- **Error**: `Failed to resolve import "*.graphql"` during `npm run build`
- **Root cause**: Vite does not natively handle `.graphql` file imports. The dev server worked because of a Babel plugin in the dev config, but the production build used a different pipeline that lacked the plugin.
- **Fix**: Added `vite-plugin-graphql` to `vite.config.ts` plugins array and moved the GraphQL transform from Babel config to the Vite plugin.
- **Prevention**: When adding file-type imports beyond JS/TS/CSS/JSON, always configure BOTH dev and production build pipelines. Check `npm run build` after adding any new import type.
- **Hits**: 3
```

---

## 7. Lesson File Structure

The `.claude/lessons.md` file uses this structure:

```markdown
# Lessons Learned

> Auto-generated by learn-by-mistake. Do not remove the section headers.

## Active Lessons
<!-- Promoted lessons applied preventively. Max 50. Sorted by Hits descending. -->

## Pending
<!-- First-occurrence errors awaiting confirmation. Max 20. -->

## Archive
<!-- Pruned or demoted lessons kept for reference. -->
```

### Rules:
- **Active Lessons**: Max 50 entries. These are checked against every error.
- **Pending**: Max 20 entries. Awaiting second occurrence to promote.
- **Archive**: No hard limit. Low-priority reference material.
- Always maintain the three section headers even if sections are empty.
- Sort Active Lessons by Hits count (highest first) when rewriting the file.

---

## 8. Pruning Rules

The lesson file must stay lean to remain useful. Apply these rules automatically:

### When Active Lessons reaches 50:
1. Find the lesson with the lowest Hits count AND the oldest date.
2. Move it to `## Archive`.
3. Then add the new lesson to Active.

### When Pending reaches 20:
1. Remove the oldest pending lesson (it never recurred — likely a one-off).
2. Then add the new pending entry.

### `/forget <number>` command:
- Remove the lesson at the specified position (1-indexed within its section).
- Tell the user which lesson was removed.

### `/lessons prune` command:
- Archive all Active lessons with `Hits: 0` that are older than 30 days.
- Remove all Pending lessons older than 30 days.
- Report how many lessons were pruned.

### `/lessons` command (no arguments):
- Display a summary: count of active, pending, and archived lessons.
- List the top 5 most-hit active lessons with their one-line summaries.

---

## 9. Transparency Protocol

Always communicate lesson activity to the user. Never operate silently.

| Event | Message |
|-------|---------|
| Lesson applied | "Applying lesson from [date]: [summary]" |
| New pending lesson | "New lesson learned: [summary]" |
| Lesson promoted | "Lesson confirmed (2nd occurrence): [summary]" |
| Lesson archived | "Archived lesson: [summary] (low usage)" |
| Lesson file created | "Created `.claude/lessons.md` — error memory initialized." |
| Lesson file loaded | No message (silent load at session start) |
| Secret redacted | "Redacted potential secret from lesson before saving." |

---

## 10. Error Handling and Edge Cases

### `.claude/lessons.md` does not exist
- Create it from the template (Section 7) on the first error that qualifies for a lesson.
- Create the `.claude/` directory if it does not exist.
- Tell the user: "Created `.claude/lessons.md` — error memory initialized."

### `.claude/lessons.md` is corrupted or unparseable
- Rename it to `.claude/lessons.md.bak` with a timestamp suffix.
- Create a fresh file from the template.
- Tell the user: "Lesson file was corrupted. Backed up to `.claude/lessons.md.bak` and created a fresh file."

### Lesson extraction must NEVER block the workflow
- Fixing the error is always the priority. Lesson extraction happens after the fix is confirmed.
- If lesson writing fails for any reason (permissions, disk, etc.), warn the user and move on.
- Never ask the user to pause work for lesson management.

### Conflicting lessons
- If two active lessons give contradictory advice for the same error pattern, flag it to the user.
- Ask which lesson to keep. Archive the other.

### Duplicate detection
- Before writing a new lesson, check if a substantially similar lesson already exists (same category + similar error pattern).
- If found, update the existing lesson's Hits count instead of creating a duplicate.
- If the new fix is better, update the existing lesson's Fix and Prevention fields.

---

## 11. Slash Commands Reference

| Command | Action |
|---------|--------|
| `/learn` | Force-create or force-promote a lesson from the last error |
| `/lessons` | Show summary of all lessons (counts + top 5) |
| `/lessons prune` | Archive stale lessons, clean up pending |
| `/forget <n>` | Remove lesson number N from active list |
| `/lessons search <term>` | Search lessons by keyword |

### `/learn` behavior
- If there was a recent error in this session: extract a lesson and add it directly to Active (skip Pending).
- If no recent error: prompt the user — "What lesson would you like to record?" — then format their response as a lesson entry.

### `/lessons search <term>` behavior
- Search all sections (Active, Pending, Archive) for lessons matching the term.
- Match against summary, error pattern, root cause, and prevention fields.
- Display matching lessons with their section and position number.

---

## 12. Integration Notes

### With version control
- `.claude/lessons.md` should be committed to the repository so lessons persist across machines and team members.
- Add `.claude/lessons.md.bak` to `.gitignore` (backup files should not be committed).

### With CI/CD
- Lessons are project-scoped. Each project has its own `.claude/lessons.md`.
- Lessons learned in CI failures can be manually added via `/learn`.

### With other skills
- This skill is passive — it activates on error events, not on explicit invocation.
- It cooperates with any other debugging or development skill.
- It does not interfere with error output or test runners.
