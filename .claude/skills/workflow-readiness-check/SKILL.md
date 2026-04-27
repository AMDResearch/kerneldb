---
name: workflow-readiness-check
description: |
  Gate check before promoting a draft workflow to active. Use after
  workflow-create to verify the packet is complete enough for autonomous
  execution. Validates dossier sections, checks for write-scope conflicts,
  and produces a pass/fail readiness checklist.
---

# workflow-readiness-check

Canonical instructions live in `.agents/skills/workflow-readiness-check/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
