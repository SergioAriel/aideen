---
name: forget
description: "Remove or archive a specific lesson. Usage: /forget <lesson-number-or-search-term>"
---

# /forget — Remove or Archive a Lesson

You are the Learn-by-Mistake skill's lesson removal tool. The user has triggered `/forget` to remove or archive a specific lesson from `.claude/lessons.md`.

## Instructions

1. **Parse the argument**:
   - If a number is given (e.g., `/forget 5`): find lesson #5 by its number
   - If text is given (e.g., `/forget docker volumes`): search for matching lessons by keyword across summary, error, root cause, fix, and prevention fields
   - If no argument is given: ask the user which lesson to forget

2. **Find the lesson** in `.claude/lessons.md`:
   - Search in Active Lessons first, then Pending
   - If multiple matches found (keyword search), list them and ask the user to pick one
   - If no match found: "No lesson found matching '[argument]'. Use `/lessons` to see all lessons."

3. **Show the lesson and confirm**:
   ```
   Found lesson to archive:

   ### [2026-03-10] config: Mount Docker volumes with correct permissions
   - **Error**: Permission denied errors inside container when writing to mounted volume
   - **Root cause**: Container runs as root (UID 0) but host volume is owned by a non-root user
   - **Fix**: Use `--user $(id -u):$(id -g)` flag or match UID/GID with host
   - **Prevention**: Always set user/group mapping when mounting host directories into containers
   - **Hits**: 3

   Archive this lesson? (yes/no/delete)
   ```

4. **On "yes" (default)**: Move the lesson to the Archive section
   - Add `**Archived**: [YYYY-MM-DD]` to the lesson
   - Remove it from its current section
   - Confirm: "Lesson #5 archived. It won't be applied automatically but can be restored with `/lessons search`."

5. **On "delete"**: Remove the lesson entirely
   - Delete it from the file completely
   - Confirm: "Lesson #5 permanently deleted."
   - Warn: "This cannot be undone."

6. **On "no"**: Cancel and do nothing.

## Important

- Archived lessons are NOT deleted — they remain searchable via `/lessons search`
- When moving to Archive, preserve the original lesson number for reference
- If the user forgets a lesson that has high hits (>5), warn them: "This lesson has been applied [N] times. Are you sure you want to archive it?"
