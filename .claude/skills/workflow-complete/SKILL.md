---
name: workflow-complete
description: |
  Perform the full completion sequence for a finished workflow. Use when all
  acceptance criteria are met and the workflow is ready to move to done state.
  Updates dossier, writes final run-log and handoff entries, moves the packet
  directory, and archives the entry in active-workflows.md.
---

# workflow-complete

Canonical instructions live in `.agents/skills/workflow-complete/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
