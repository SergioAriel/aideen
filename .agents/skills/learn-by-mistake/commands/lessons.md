---
name: lessons
description: "Browse, search, and manage learned lessons. Usage: /lessons [search|stats|prune|export]"
---

# /lessons — Browse, Search & Manage Lessons

You are the Learn-by-Mistake skill's lesson manager. The user has triggered `/lessons` to interact with their learned lessons stored in `.claude/lessons.md`.

## Sub-commands

Parse the user's input after `/lessons` to determine which sub-command to run. If no argument is given, default to showing all lessons.

### `/lessons` (no arguments) — Show All Lessons

1. Read `.claude/lessons.md`
2. Display all **Active Lessons** grouped by category
3. Format as a clean table or grouped list:
   ```
   ## Active Lessons (14 total)

   ### syntax (3)
   1. Quote variables in bash conditionals
   5. Use double brackets [[ ]] for string comparison
   9. Escape special chars in sed patterns

   ### config (2)
   3. Set NODE_ENV before running build
   7. Include trailing slash in API base URLs

   ... etc
   ```
4. Also show count of Pending and Archived lessons at the bottom

### `/lessons search <term>` — Search Lessons

1. Search across all sections (Active, Pending, Archive) in `.claude/lessons.md`
2. Match against: summary, error, root cause, fix, and prevention fields
3. Case-insensitive search
4. Display matching lessons with their section indicated:
   ```
   Found 3 lessons matching "docker":

   [Active] #4. Mount volumes with correct permissions
   [Active] #11. Use BuildKit for multi-stage builds
   [Archive] #2. Pin base image versions
   ```

### `/lessons stats` — Show Statistics

1. Read `.claude/lessons.md` and compute:
   - Total lessons (active + pending + archived)
   - Breakdown by section (Active / Pending / Archive)
   - Breakdown by category
   - Top 5 most-hit lessons (highest Hits count)
   - 5 most recently added lessons
   - Lessons with 0 hits (candidates for pruning)
2. Display as a formatted summary

### `/lessons prune` — Archive Stale Lessons

1. Find Active lessons where:
   - **Hits** is 0, AND
   - **Added** date is older than 30 days
2. List the candidates and ask user for confirmation
3. On approval, move each lesson to the Archive section
4. Update the lesson format to include `**Archived**: [YYYY-MM-DD]`

### `/lessons export` — Export Shareable Lessons

1. Read all Active lessons from `.claude/lessons.md`
2. Output them in a clean, shareable format:
   - Remove hit counts and dates
   - Remove any project-specific paths or sensitive information
   - Format as a standalone markdown block
3. Example output:
   ```markdown
   # Lessons Learned (exported)

   ## syntax
   - **Quote variables in bash conditionals**: Use `"$var"` not `$var` in `[[ ]]` tests
   - **Use double brackets for string comparison**: `[[ ]]` handles spaces; `[ ]` does not

   ## config
   - **Set NODE_ENV before build**: Export NODE_ENV=production before running build scripts
   ```

## Error Handling

- If `.claude/lessons.md` doesn't exist: "No lessons file found. Lessons are created automatically when errors occur, or use `/learn` to manually extract one."
- If a sub-command is not recognized: Show the usage help with available sub-commands.
