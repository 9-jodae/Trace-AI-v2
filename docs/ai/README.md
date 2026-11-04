# AI harness map

Read `AGENTS.md` first. Then read the workflow for the task and any relevant rule in this directory. Consult the Korean `docs/` files for product context and detailed procedures.

Priority: explicit user request, `AGENTS.md`, applicable `docs/ai/` workflow or rule, then tool-specific wrapper. Edit shared rules only in `AGENTS.md` or `docs/ai/`; keep `.claude/skills/` and `.codex/prompts/` thin.

| Claude skill | Codex skill | Manual Codex prompt | Shared workflow |
| --- | --- | --- | --- |
| `verify` | `verify` | `verify-changes.md` | `workflows/verify.md` |
| `commit-and-pr` | `commit-and-pr` | `commit-and-pr.md` | `workflows/commit-and-pr.md` |
| `review-response` | `review-response` | `review-response.md` | `workflows/review-response.md` |

| Task | Workflow | Rules |
| --- | --- | --- |
| Change server code | `workflows/verify.md` | `rules/python-fastapi.md`, `rules/testing.md`, `rules/security.md` |
| Verify changes | `workflows/verify.md` | `rules/testing.md` |
| Commit and open a PR | `workflows/commit-and-pr.md` | `rules/security.md` |
| Respond to PR review | `workflows/review-response.md` | Applicable rules above |

Keep this harness in English. Keep human documentation, user-facing text, commits, PR bodies, and reports in Korean, as specified in `AGENTS.md`.
