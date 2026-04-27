---
name: state-check
description: |
  Verify that all workflow and project state documents are consistent. Use when
  the user says "state check", "check state", or when session-init flags
  inconsistencies. Checks for orphan directories, dangling table entries,
  lifecycle mismatches, stale handoffs, and empty run-logs.
---

# state-check

Canonical instructions live in `.agents/skills/state-check/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
