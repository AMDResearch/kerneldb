---
name: session-init
description: |
  Bootstrap a new agent session. Use at the start of every conversation or when
  the user says "start session", "init", or "begin". Reads project metadata,
  identifies active workflows, detects prior aborted sessions, and produces a
  concise briefing so the agent can begin work with minimal context loading.
---

# session-init

Canonical instructions live in `.agents/skills/session-init/SKILL.md`.

Use that file as the source of truth. This wrapper exists only so Claude-facing skill discovery can find the installed skill.
