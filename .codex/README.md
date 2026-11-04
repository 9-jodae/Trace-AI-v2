# Codex in this repository

Codex reads `AGENTS.md` as repository instructions. Read `docs/ai/README.md` for the shared workflow map. Use the paired files in `.codex/prompts/` as manual task prompts: refer to or paste the relevant file when starting a task.

This repository uses thin Codex skills in `.agents/skills/` and needs no project settings. The shared procedures live in `docs/ai/`.

| Claude skill | Codex skill | Manual Codex prompt | Shared workflow |
| --- | --- | --- | --- |
| `verify` | `verify` | `verify-changes.md` | `docs/ai/workflows/verify.md` |
| `commit-and-pr` | `commit-and-pr` | `commit-and-pr.md` | `docs/ai/workflows/commit-and-pr.md` |
| `review-response` | `review-response` | `review-response.md` | `docs/ai/workflows/review-response.md` |
