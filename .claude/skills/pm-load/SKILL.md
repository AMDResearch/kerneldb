---
name: pm-load
description: |
  Load the smallest set of Project Memory units relevant to the current task.
  Use at the start of any task that needs project context. Reads pm-index.md,
  selects 1-3 relevant units based on task scope, and loads them into working
  memory without polluting context with unrelated knowledge.
---

# pm-load

Canonical instructions live in `.agents/skills/pm-load/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
