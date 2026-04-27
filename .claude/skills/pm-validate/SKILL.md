---
name: pm-validate
description: |
  Validate Project Memory for structural correctness. Use when the user says
  "validate PM", when session-init flags inconsistencies, or periodically to
  detect PM rot. Checks for broken anchors, stale units, schema violations,
  orphan files, and index drift. Auto-heals index entries.
---

# pm-validate

Canonical instructions live in `.agents/skills/pm-validate/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
