#!/bin/bash
# detect-error.sh — PostToolUse hook for Bash tool
# Reads tool result JSON from stdin, checks for errors, and emits
# a systemMessage telling Claude to consult lessons before fixing.

set -euo pipefail

# Read the full stdin (PostToolUse provides JSON with tool result)
INPUT=$(cat 2>/dev/null || echo "")

# Bail out silently if no input
if [ -z "$INPUT" ]; then
  echo '{}'
  exit 0
fi

# Extract exit_code from the tool result JSON
# Expected shape: { "tool_name": "Bash", "tool_input": {...}, "tool_result": { "exit_code": N, "stdout": "...", "stderr": "..." } }
EXIT_CODE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Handle different possible JSON structures
    result = data.get('tool_result', data)
    if isinstance(result, dict):
        print(result.get('exit_code', 0))
    else:
        print(0)
except:
    print(0)
" 2>/dev/null || echo "0")

# Extract stderr
STDERR=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    result = data.get('tool_result', data)
    if isinstance(result, dict):
        print(result.get('stderr', ''))
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")

# Also check stdout for error patterns (some tools report errors in stdout)
STDOUT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    result = data.get('tool_result', data)
    if isinstance(result, dict):
        print(result.get('stdout', ''))
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")

ERROR_DETECTED=false
ERROR_TYPE=""

# Check 1: Non-zero exit code
if [ "$EXIT_CODE" != "0" ] && [ -n "$EXIT_CODE" ]; then
  ERROR_DETECTED=true
  ERROR_TYPE="exit_code=$EXIT_CODE"
fi

# Exclusion patterns — false-positive strings that should NOT trigger error detection
EXCLUSION_PATTERNS="0 error|no error|0 failed|error handling|error\.ts|error\.js"

# Helper: check if text matches exclusion patterns (returns 0 if should be excluded)
should_exclude() {
  local text="$1"
  echo "$text" | grep -iqE "$EXCLUSION_PATTERNS" 2>/dev/null
}

# Check 2: Error patterns in stderr
if [ -n "$STDERR" ]; then
  # Case-sensitive patterns (strong signals — trigger regardless of exit code)
  CASE_SENSITIVE_PATTERNS="FAIL|ENOENT|TypeError|SyntaxError|ReferenceError|BUILD FAILED|Traceback|Permission denied|Cannot find"
  # Case-insensitive patterns (weaker signals — only trigger when exit code is also non-zero)
  CASE_INSENSITIVE_PATTERNS="error|failed|panic|not found|compilation failed"

  if ! should_exclude "$STDERR"; then
    if echo "$STDERR" | grep -qE "$CASE_SENSITIVE_PATTERNS" 2>/dev/null; then
      ERROR_DETECTED=true
      MATCH=$(echo "$STDERR" | grep -oE "$CASE_SENSITIVE_PATTERNS" 2>/dev/null | head -1)
      if [ -n "$ERROR_TYPE" ]; then
        ERROR_TYPE="$ERROR_TYPE, stderr=$MATCH"
      else
        ERROR_TYPE="stderr=$MATCH"
      fi
    elif [ "$EXIT_CODE" != "0" ] && [ -n "$EXIT_CODE" ] && echo "$STDERR" | grep -iqE "$CASE_INSENSITIVE_PATTERNS" 2>/dev/null; then
      ERROR_DETECTED=true
      MATCH=$(echo "$STDERR" | grep -ioE "$CASE_INSENSITIVE_PATTERNS" 2>/dev/null | head -1)
      if [ -n "$ERROR_TYPE" ]; then
        ERROR_TYPE="$ERROR_TYPE, stderr=$MATCH"
      else
        ERROR_TYPE="stderr=$MATCH"
      fi
    fi
  fi
fi

# Check 3: Error patterns in stdout (some commands write errors to stdout)
if [ "$ERROR_DETECTED" = false ] && [ -n "$STDOUT" ]; then
  CASE_SENSITIVE_PATTERNS="FAIL|ENOENT|TypeError|SyntaxError|ReferenceError|BUILD FAILED|Traceback|Permission denied|Cannot find"
  CASE_INSENSITIVE_PATTERNS="error|failed|panic|not found|compilation failed"

  if ! should_exclude "$STDOUT"; then
    if echo "$STDOUT" | grep -qE "$CASE_SENSITIVE_PATTERNS" 2>/dev/null; then
      ERROR_DETECTED=true
      MATCH=$(echo "$STDOUT" | grep -oE "$CASE_SENSITIVE_PATTERNS" 2>/dev/null | head -1)
      ERROR_TYPE="stdout=$MATCH"
    elif [ "$EXIT_CODE" != "0" ] && [ -n "$EXIT_CODE" ] && echo "$STDOUT" | grep -iqE "$CASE_INSENSITIVE_PATTERNS" 2>/dev/null; then
      ERROR_DETECTED=true
      MATCH=$(echo "$STDOUT" | grep -ioE "$CASE_INSENSITIVE_PATTERNS" 2>/dev/null | head -1)
      ERROR_TYPE="stdout=$MATCH"
    fi
  fi
fi

# Emit result
if [ "$ERROR_DETECTED" = true ]; then
  # Escape for JSON
  ERROR_TYPE_ESCAPED=$(echo "$ERROR_TYPE" | sed 's/"/\\"/g' | tr -d '\n')
  cat <<EOF
{"systemMessage": "Error detected [$ERROR_TYPE_ESCAPED]. Check .claude/lessons.md for matching lessons before fixing. After resolving, evaluate if this warrants a new lesson per SKILL.md rules (2 occurrences = confidence gate, or user can force with /learn)."}
EOF
else
  echo '{}'
fi
