# Guardrails

## Workspace Boundaries

- Work only inside the current repository unless the user explicitly asks otherwise.
- Treat `.agents/` as tracked project infrastructure.
- Treat `.untracked/` as local review and scratch space.

## Git Discipline

- Commit meaningful units of work with descriptive messages.
- Do not force-push, rebase published branches, or reset shared history without explicit user approval.
- Do not commit secrets, credentials, or large binary files.
- Do not delete branches that other workflows or agents may depend on.
- When in doubt about a destructive git operation, ask the user first.

## Scope Discipline

- Use workflow packets for substantial work.
- Preserve contract-controlled sections after a workflow becomes active.
- Stop when the requested outcome cannot be met under the current contract.
- Before running in parallel with other agents, check `.agents/state/active-workflows.md` and avoid uncoordinated write-scope overlap.
- Do not expand scope beyond what the dossier specifies. Note proposed expansions in `handoff.md` under `Proposed Spec Changes`.

## Context Hygiene

- Load only the PM units, workflow files, and documentation relevant to the current task.
- Do not front-load the entire PM index, all workflow packets, or all skill docs at session start.
- Use `pm-load` to select the minimal relevant unit set.
- When resuming a workflow, read `handoff.md` and `dossier.md` first. Read `run-log.md` only when execution history is specifically needed.
- If context is growing too large, summarize what you know and drop the source material rather than carrying everything forward.

## Verification Expectations

- Verify meaningful changes by running relevant tests, linters, or manual checks.
- Record verification outcomes in the workflow `run-log.md` or session capture.
- If work is unverified, say so explicitly in `handoff.md` and `run-log.md`.
- Do not mark acceptance criteria as met without verification evidence.

## Delegation Guidance

- When delegating work to a sub-agent or parallel session, provide the workflow packet as the contract. Do not rely on verbal instructions alone.
- Each delegated unit of work should have its own workflow packet with a distinct write scope.
- The delegating agent is responsible for checking `active-workflows.md` for overlap before assigning parallel work.
- Sub-agents follow the same guardrails as the primary agent.

## PM Expectations

- Update PM only with durable project knowledge.
- Do not turn PM into a transcript archive.
- Filter for durability before writing: will a fresh session 2 weeks from now benefit from this?
- Update `pm-current-state.md` when the project-wide situation changes materially.

## State Persistence

- Update `handoff.md` and `run-log.md` incrementally after every significant step, not at end-of-session.
- If you have completed 3 or more meaningful steps without updating `handoff.md` or `run-log.md`, stop current work and update state files before continuing.
- If you estimate you are approaching context limits, immediately update `handoff.md` and `run-log.md` before continuing.

## Checkpoint Protocol

Checkpoints are lightweight state saves — update `handoff.md` with current status and append
an entry to `run-log.md`. Each checkpoint should take under one minute. Checkpoint at these
triggers:

- After completing (or determining you cannot complete) an acceptance criterion.
- After creating or modifying more than three files since the last checkpoint.
- After running verification (tests, linters, manual checks) regardless of outcome.
- Before any operation you expect to take more than five minutes.
- When context usage is above ~75% — checkpoint before the system compresses prior messages.

Checkpoints supplement the incremental persistence rules in State Persistence above. They
exist so that if the session ends unexpectedly, the next session can pick up from the last
checkpoint rather than replaying the entire session's work.

## Stop Conditions

- **Contract violation**: If you realize you are changing acceptance criteria, scope, or failure policy without user approval, stop immediately and report.
- **Unresolvable blocker**: If a dependency cannot be met and the failure policy is `stop`, move the workflow to `failed/` and report.
- **Write-scope collision**: If your intended write scope collides with another active workflow without explicit coordination, stop and ask the user.
- **Safety risk**: If the requested work could cause data loss, security vulnerabilities, or irreversible damage, stop and confirm with the user.
- **Uncertainty**: If you are unsure whether an action is within scope or safe, stop and ask rather than proceeding.
