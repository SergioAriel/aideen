#!/bin/bash
# load-lessons.sh — SessionStart hook
# Reads .claude/lessons.md at session start and emits a summary
# so Claude is aware of accumulated lessons from the beginning.

set -euo pipefail

LESSONS_FILE=".claude/lessons.md"

# Check if lessons file exists
if [ ! -f "$LESSONS_FILE" ]; then
  cat <<'EOF'
{"systemMessage": "Learn-by-Mistake skill active. No lessons file found yet (.claude/lessons.md). Lessons will be created as errors are encountered. The user can also run /learn to force-extract a lesson from any error."}
EOF
  exit 0
fi

# Parse lesson counts and categories in a single python3 call
read -r ACTIVE_SECTION_COUNT PENDING_COUNT ARCHIVE_COUNT CATEGORIES <<< $(python3 -c "
import re
from collections import Counter

try:
    with open('$LESSONS_FILE', 'r') as f:
        content = f.read()

    sections = re.split(r'^## ', content, flags=re.MULTILINE)
    active = pending = archive = 0
    active_text = ''

    for section in sections:
        lesson_count = len(re.findall(r'^### \[', section, re.MULTILINE))
        header = section.split('\n')[0].strip().lower()
        if 'active' in header:
            active = lesson_count
            active_text = section
        elif 'pending' in header:
            pending = lesson_count
        elif 'archive' in header:
            archive = lesson_count

    # Extract categories from heading lines: ### [YYYY-MM-DD] category: summary
    cats = re.findall(r'^### \[[\d-]+\]\s+(\w+):', active_text, re.MULTILINE)
    counts = Counter(cats)
    cat_str = ', '.join(f'{cat}({n})' for cat, n in counts.most_common()) if cats else 'none'

    print(f'{active} {pending} {archive} {cat_str}')
except:
    print('0 0 0 none')
" 2>/dev/null || echo "0 0 0 none")

# Build the summary message
SUMMARY="Learn-by-Mistake skill active. Loaded $ACTIVE_SECTION_COUNT active lessons"

if [ "$PENDING_COUNT" -gt 0 ] 2>/dev/null; then
  SUMMARY="$SUMMARY, $PENDING_COUNT pending"
fi

if [ "$ARCHIVE_COUNT" -gt 0 ] 2>/dev/null; then
  SUMMARY="$SUMMARY, $ARCHIVE_COUNT archived"
fi

SUMMARY="$SUMMARY. Categories: $CATEGORIES."
SUMMARY="$SUMMARY When you encounter errors, check these lessons BEFORE attempting a fix. Consult .claude/lessons.md for full details."

# Escape for JSON
SUMMARY_ESCAPED=$(echo "$SUMMARY" | sed 's/"/\\"/g' | tr -d '\n')

cat <<EOF
{"systemMessage": "$SUMMARY_ESCAPED"}
EOF
