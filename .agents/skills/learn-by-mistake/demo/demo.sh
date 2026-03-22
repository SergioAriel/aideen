#!/usr/bin/env bash
# Demo script for learn-by-mistake skill
# Simulates the 3-session error-learning flow

BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
MAGENTA='\033[35m'
RESET='\033[0m'

clear

echo -e "${BOLD}${CYAN}learn-by-mistake${RESET}${DIM} — persistent error memory for Claude Code${RESET}"
echo ""
sleep 0.3

# ── Session 1 ──────────────────────────────────────────────
echo -e "${BOLD}${CYAN}━━━ Session 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.2

echo -e "  ${BOLD}\$ npm test${RESET}"
sleep 0.3
echo -e "  ${RED}FAIL${RESET} src/auth.test.ts"
echo -e "  ${RED}TypeError: Cannot read properties of undefined (reading 'token')${RESET}"
sleep 0.3

echo ""
echo -e "  ${DIM}Analyzing error...${RESET}"
sleep 0.2
echo -e "  ${DIM}Extracting root cause...${RESET}"
sleep 0.2
echo -e "  ${DIM}Checking lessons.md — no match found${RESET}"
sleep 0.2

echo ""
echo -e "  ${YELLOW}💡 Pending lesson:${RESET}"
echo -e "  ${YELLOW}   [test] Always initialize auth context in beforeEach${RESET}"
echo ""
sleep 0.4

# ── Session 2 ──────────────────────────────────────────────
echo -e "${BOLD}${CYAN}━━━ Session 2 (days later) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.2

echo -e "  ${BOLD}\$ npm test${RESET}"
sleep 0.3
echo -e "  ${RED}FAIL${RESET} src/payments.test.ts"
echo -e "  ${RED}TypeError: Cannot read properties of undefined (reading 'session')${RESET}"
sleep 0.3

echo ""
echo -e "  ${DIM}Analyzing error...${RESET}"
sleep 0.2
echo -e "  ${DIM}Matching root cause to pending lessons...${RESET}"
sleep 0.2

echo ""
echo -e "  ${GREEN}✅ Lesson confirmed (2nd occurrence)${RESET}"
echo -e "  ${GREEN}📝 Active lesson #12:${RESET}"
echo -e "  ${GREEN}   [test] Initialize context objects (auth, session, user)${RESET}"
echo -e "  ${GREEN}   in beforeEach, not in individual tests${RESET}"
echo ""
sleep 0.4

# ── Session 3 ──────────────────────────────────────────────
echo -e "${BOLD}${CYAN}━━━ Session 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.2

echo -e "  ${DIM}Writing new test...${RESET}"
sleep 0.3

echo ""
echo -e "  ${GREEN}📝 Applying lesson #12:${RESET} Initialize context in beforeEach"
sleep 0.2
echo -e "  ${DIM}   → Adding beforeEach with auth, session, user context${RESET}"
sleep 0.2
echo -e "  ${GREEN}   → Test passes first time ✓${RESET}"
echo ""
sleep 0.3

echo -e "${DIM}No prompting. No reminders. The skill remembers so you don't have to.${RESET}"
echo ""
