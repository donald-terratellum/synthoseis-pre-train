# Agentic Session Wrap-Up Workflow

Use this prompt as-is in Copilot Chat when I want a complete end-of-session wrap-up.

## Objective
Perform a full session closeout by:
1. creating/updating a detailed session summary in `.agentic-docs`,
2. preparing clean, logical git commits,
3. offering the next GitHub step (push and/or PR).

## Required Workflow

### 1) Discover and analyze context
- Inspect existing summaries in `.agentic-docs` and infer their:
  - section structure,
  - technical depth,
  - writing style,
  - verbosity level,
  - chronology/detail expectations.
- Inspect chat/session context from this coding session to identify:
  - goals,
  - decisions,
  - changes made,
  - debugging performed,
  - tests run,
  - unresolved items.
- Inspect git changes since the latest summary point:
  - `git status --short`
  - `git diff --stat`
  - `git log --oneline --decorate -n 20`
  - `git diff` (scoped where needed)
- Correlate chat actions with concrete file changes and commits.

### 2) Produce session summary in `.agentic-docs`
- Create a new summary file in `.agentic-docs` (or update the most recent one if explicitly better).
- Use existing `.agentic-docs` summaries as the style/template source of truth.
- Include, at minimum, these sections (adapting names to local style):
  - Context
  - Goals
  - Work Completed
  - Implementation Details
  - Validation / Tests
  - Git Changes
  - Open Questions / Risks
  - Next Steps
  - Timestamp and Author
- Explicitly capture:
  - what was done,
  - how it was done,
  - when it was done,
  - who did it (human, copilot, or both where applicable).
- When referencing code changes, include key files and short rationale per file.
- Keep claims traceable to chat context and/or git evidence.

### 3) Prepare commits
- Review changed files and group them into logical commit units.
- Prefer multiple commits when it improves readability/reviewability.
- Use clear commit messages (Conventional Commit style is preferred), e.g.:
  - `feat: add sliding window stats loss components`
  - `fix: correct pre-head polarity gating default`
  - `test: add coverage for sliding stats MAE/MSE terms`
  - `docs: add session summary for YYYY-MM-DD`
- Before committing, show a concise proposed commit plan (files per commit + message).
- Then create the commit(s).

### 4) Offer GitHub next action
- Detect whether remotes are configured (`git remote -v`).
- If a remote exists, offer to:
  - push current branch, and/or
  - open/create a PR with a draft or final description.
- If no remote exists, say so and provide the minimal command needed to add one.

## Output Quality Bar
- Be precise and evidence-based.
- Do not invent actions, tests, timestamps, or commits.
- Keep summary technical and useful for future resumption.
- Make commits easy for a reviewer to scan and understand.

## Execution Constraints
- Do not revert unrelated pre-existing changes.
- Avoid destructive git operations.
- Ask for confirmation before pushing or creating PRs.

## Optional Enhancements
- If unsure about summary filename convention, infer from existing `.agentic-docs` names and follow that pattern.
- If there are no meaningful code changes, still produce/update summary and skip commit creation with explanation.
