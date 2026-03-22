#!/bin/bash
# preserve-lessons.sh — PreCompact hook
# Before context compaction, emits a systemMessage reminding Claude
# to preserve lesson awareness across the compacted context.

set -euo pipefail

LESSONS_FILE=".claude/lessons.md"

# Check if lessons file exists
if [ ! -f "$LESSONS_FILE" ]; then
  cat <<'EOF'
{"systemMessage": "CONTEXT COMPACTION NOTICE: Learn-by-Mistake skill is active but no lessons file exists yet. No lesson state to preserve."}
EOF
  exit 0
fi

# Count active lessons for the reminder
ACTIVE_COUNT=$(grep -c '^### \[' "$LESSONS_FILE" 2>/dev/null || echo "0")

# Extract lesson summaries for a quick reference list
TITLES=$(python3 -c "
import re

try:
    with open('$LESSONS_FILE', 'r') as f:
        content = f.read()

    # Find Active Lessons section
    active_match = re.search(r'## Active Lessons\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if active_match:
        titles = re.findall(r'^### \[.+\] .+: (.+)$', active_match.group(1), re.MULTILINE)
        for t in titles[:10]:  # Cap at 10 to keep message short
            print(f'  - {t}')
        if len(titles) > 10:
            print(f'  ... and {len(titles) - 10} more')
    else:
        print('  (none)')
except:
    print('  (unable to parse)')
" 2>/dev/null || echo "  (unable to parse)")

# Build preservation message
MSG="CONTEXT COMPACTION — PRESERVE LESSON AWARENESS: The learn-by-mistake skill has $ACTIVE_COUNT active lessons in .claude/lessons.md. After compaction, continue to check lessons before fixing errors. Key lessons:\n$TITLES\nAlways read .claude/lessons.md when encountering errors."

# Escape for JSON (handle newlines and quotes)
MSG_ESCAPED=$(echo "$MSG" | python3 -c "
import sys, json
print(json.dumps(sys.stdin.read().strip())[1:-1])
" 2>/dev/null || echo "$MSG" | sed 's/"/\\"/g' | tr '\n' ' ')

cat <<EOF
{"systemMessage": "$MSG_ESCAPED"}
EOF
