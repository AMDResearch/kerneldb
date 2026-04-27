---
name: session-close
description: |
  Perform the full end-of-session sequence. Use when the user says "close",
  "wrap up", "done for now", "session-close", or similar. Updates workflow
  documents, persists PM knowledge, commits changes in logical groups, runs
  session-capture, and delivers a summary.
---

# session-close

Canonical instructions live in `.agents/skills/session-close/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
