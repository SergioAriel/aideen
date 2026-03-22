---
name: setup
description: "Install learn-by-mistake hooks into your project or user config. Merges with existing hooks without conflicts. Usage: /learn setup [--global]"
---

# /learn setup

Install the learn-by-mistake hooks for automatic error detection. This command MERGES hooks into your existing configuration — it never overwrites other hooks.

## What it does

1. Detect if hooks should go to project level (`.claude/hooks.json`) or user level (`~/.claude/hooks.json`)
2. Read existing hooks file (if any)
3. Merge learn-by-mistake hooks into the existing structure
4. Write the merged result
5. Verify installation

## Protocol

### Step 1: Determine scope

- If the user said `/learn setup --global` → use `~/.claude/hooks.json`
- Otherwise → use `.claude/hooks.json` in the current project root
- Tell the user which location you're using

### Step 2: Read existing hooks

Read the target hooks file. If it exists, parse the JSON. If it doesn't exist, start with:
```json
{"hooks": {}}
```

### Step 3: Merge — NEVER overwrite

For each hook type (PostToolUse, SessionStart, PreCompact):

**If the hook type doesn't exist** in the target file: add it entirely.

**If the hook type already exists**: APPEND learn-by-mistake hooks to the existing array. Never remove or modify existing hooks.

To prevent duplicates, check if any existing hook command contains "learn-by-mistake" or "detect-error" before adding.

### Step 4: Determine script paths

The hook scripts live in the skill directory. Find them:
```bash
# Check common locations
SKILL_DIR=""
for dir in \
  ~/.claude/skills/learn-by-mistake \
  .claude/skills/learn-by-mistake \
  node_modules/learn-by-mistake-skill; do
  if [ -f "$dir/hooks/scripts/detect-error.sh" ]; then
    SKILL_DIR="$dir"
    break
  fi
done
```

Use ABSOLUTE paths in the hooks.json to avoid working directory issues.

### Step 5: Write the merged hooks

The learn-by-mistake hooks to merge:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash <SKILL_DIR>/hooks/scripts/detect-error.sh",
            "timeout": 5
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash <SKILL_DIR>/hooks/scripts/load-lessons.sh",
            "timeout": 3
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash <SKILL_DIR>/hooks/scripts/preserve-lessons.sh",
            "timeout": 3
          }
        ]
      }
    ]
  }
}
```

Replace `<SKILL_DIR>` with the absolute path found in Step 4.

### Step 6: Verify

After writing, read back the file and confirm:
- PostToolUse has a learn-by-mistake entry
- SessionStart has a learn-by-mistake entry
- PreCompact has a learn-by-mistake entry
- Existing hooks from OTHER tools are still present and unmodified

### Step 7: Report

Tell the user:
```
✅ learn-by-mistake hooks installed at <path>
  - PostToolUse: error detection on Bash commands
  - SessionStart: lesson loading at session start
  - PreCompact: lesson preservation during compaction

  Existing hooks preserved: <count> other hook(s) untouched.

  To uninstall: /learn uninstall
```

## Conflict Prevention

- ALWAYS read existing hooks first
- NEVER overwrite — only append
- Check for duplicates before adding
- Use absolute paths for script references
- If ANY step fails, abort and tell the user what happened — never leave a corrupted hooks.json

## Uninstall

If the user says `/learn uninstall` or `/learn setup --remove`:
1. Read the hooks file
2. Remove only entries whose command contains "learn-by-mistake" or "detect-error" or "load-lessons" or "preserve-lessons"
3. If a hook type array becomes empty after removal, remove the type key
4. If the entire hooks object becomes empty, delete the file
5. Confirm removal
