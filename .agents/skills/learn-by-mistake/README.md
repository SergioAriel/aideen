# Learn by Mistake

Your AI coding assistant makes the same mistake twice. This skill makes sure it never happens again.

**Persistent error memory for Claude Code.** Analyzes failures, extracts root-cause lessons, and applies them before the same mistake can happen again. Every error becomes institutional knowledge.

## Install

```
/plugin marketplace add JuanMarchetto/agent-skills
/plugin install learn-by-mistake@agent-skills
```

Or via [skills.sh](https://skills.sh):
```bash
npx skills add JuanMarchetto/learn-by-mistake-skill
```

Or manually:
```bash
git clone https://github.com/JuanMarchetto/learn-by-mistake-skill.git
cp -r learn-by-mistake-skill ~/.claude/skills/learn-by-mistake
```

## How It Works

![demo](demo/demo.gif)

A closed loop that turns every failure into a permanent advantage:

```
Error occurs --> Check lessons --> Fix --> Extract lesson --> Persist
                     ^                                        |
                     |                                        |
                     +----------------------------------------+
```

1. An error happens (build failure, test crash, runtime exception)
2. The skill checks `.claude/lessons.md` for a known fix
3. If found, the fix is applied immediately -- no debugging, no guessing
4. If new, the root cause is extracted into a structured lesson
5. The lesson is persisted and available for every future session

The result: mistakes that used to cost 10 minutes of debugging now cost zero.

## The Confidence Gate

Not every error deserves a permanent lesson. Typos happen. Flukes happen. The confidence gate prevents noise from polluting your knowledge base.

| Occurrence | Status | What happens |
|-----------|--------|-------------|
| 1st | **Pending** | Lesson is drafted but not yet trusted |
| 2nd | **Active** | Same root cause seen again -- lesson is confirmed and applied going forward |

First occurrence goes to **Pending**. Second occurrence promotes it to **Active**. This two-strike rule means only real, recurring patterns become lessons. One-off mistakes are filtered out naturally.

Use `/learn` to bypass the gate and force-promote a lesson when you know it matters.

## Secret Sanitization

Lessons are sanitized before they are written to disk. The skill scans every lesson for sensitive data and redacts it automatically:

- API keys and tokens --> `<REDACTED_API_KEY>`
- Passwords and secrets --> `<REDACTED_SECRET>`
- Absolute paths with usernames --> `<PROJECT_PATH>/...`
- Connection strings --> `<REDACTED_CONNECTION_STRING>`
- Environment-specific values --> `<REDACTED_ENV>`

Your lessons file is safe to commit to version control.

## Example

```
Session 1:
  $ npm test
    FAIL src/auth.test.ts
    TypeError: Cannot read properties of undefined (reading 'token')

  Lesson extracted (pending):
  [test] Always initialize auth context in beforeEach

Session 2 (days later):
  $ npm test
    FAIL src/payments.test.ts
    TypeError: Cannot read properties of undefined (reading 'session')

  Lesson confirmed (2nd occurrence):
  Active lesson #12: [test] Initialize context objects (auth, session, user)
  in beforeEach, not in individual tests

Session 3:
  User asks to write a new test...

  Applying lesson #12: Initialize context in beforeEach
  --> beforeEach block written with auth, session, and user context
  --> Test passes first time
```

No prompting. No reminders. The skill remembers so you don't have to.

It also learns from **your corrections**:

```
User: "No, we use pnpm here, not npm"

  📝 Active lesson: [correction] Use pnpm for all package operations in this project
  (User corrections skip the confidence gate — one correction is enough)

Later:
  User: "install lodash"
  📝 Applying lesson: Use pnpm
  → pnpm add lodash ✓
```

## What It Learns

| Category | Examples | Detection |
|----------|---------|-----------|
| **build** | Missing imports, webpack config errors, compilation failures | Non-zero exit from build commands |
| **test** | Uninitialized context, missing mocks, assertion mismatches | Test runner failures (jest, vitest, pytest) |
| **runtime** | Null references, unhandled promises, type coercion bugs | Exceptions in stderr |
| **types** | TypeScript strict mode violations, generic constraints | `tsc` errors, type-check failures |
| **linter** | ESLint/Prettier conflicts, rule misconfigurations | Lint command output |
| **git** | Merge conflicts, detached HEAD, hook failures | Git command stderr |
| **dependency** | Version conflicts, peer dependency warnings, missing packages | npm/yarn/pnpm errors |
| **config** | Wrong env vars, malformed JSON/YAML, path issues | Config parse errors |
| **correction** | "Don't use semicolons", "use pnpm not npm", "that's wrong" | User says something is incorrect or should be different |

**User corrections skip the confidence gate** — they go directly to Active Lessons. The user is the ground truth. One correction is enough.

## File Structure

```
learn-by-mistake-skill/
  .claude-plugin/
    plugin.json          # Skill metadata
  commands/
    learn.md             # /learn — force-extract a lesson
  hooks/
    scripts/             # Hook scripts for error detection
  references/            # Pattern databases and detection rules
  templates/             # Lesson file templates
  LICENSE
  README.md
```

At runtime, the skill reads and writes a single file in your project:

```
your-project/
  .claude/
    lessons.md           # Your project's lesson memory
```

## Two Modes

### Automatic (SKILL.md — works everywhere)
The skill is **always active** when loaded. Every time a command fails, Claude checks lessons before fixing and extracts new lessons after. No setup needed. Works on Claude Code, Codex CLI, Gemini CLI, Cursor, Windsurf.

### Accelerated (hooks — Claude Code only)
For faster detection, install the PostToolUse hooks:

```
/learn setup
```

This merges hooks into your existing `.claude/hooks.json` without touching other hooks. The hooks give you:
- Instant error detection on every Bash failure
- Lesson loading at session start
- Lesson preservation through context compaction

To remove: `/learn setup --remove`

## Commands

| Command | What it does |
|---------|-------------|
| `/learn` | Force-extract a lesson from the last error |
| `/learn setup` | Install hooks for accelerated detection (merges, never overwrites) |
| `/lessons` | Browse, search, and manage your active lessons |
| `/forget` | Remove or archive a specific lesson |

## Requirements

- **Any AI coding assistant** that supports SKILL.md
- **No external dependencies** — pure markdown, no installs, no build step
- **Optional**: Claude Code hooks for accelerated detection (`/learn setup`)

## License

[MIT](LICENSE)
