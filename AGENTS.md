# Repository instructions

This is the single source of repository rules. Keep AI harness files (`AGENTS.md`, `CLAUDE.md`, `.agents/**`, `.claude/**`, `.codex/**`, `docs/ai/**`) in English. Keep human documentation (`README.md`, other `docs/**`, issue and PR templates), user-facing messages, commit subjects, PR bodies, and reports in Korean. Put workflow details in `docs/ai/` and background in the Korean `docs/` files.

## Structure and ownership

- This is the Python AI server for docs199. Put shared code in `app/core` and feature code in `app/<feature>`. Follow `docs/ai/rules/python-fastapi.md`.
- Spring in `9-jodae/docs199-Server` owns authentication, authorization, document management, and logs. Record the agreed server contract only in that repository's `docs/ai-server-contract.md`.
- That repository's Flyway migrations are the only way to change the shared database schema. This server must not change the schema.

## Safety and verification

- Do not invent unconfirmed domains, APIs, or access rules. Record unresolved decisions in the relevant Korean `docs/` file.
- Never commit or log secrets, tokens, document contents, or presigned URLs. Follow `docs/ai/rules/security.md`.
- Update README and relevant `docs/` when behavior or setup changes. Isolate test state. Run `ruff check . && pytest` and report actual results. Follow `docs/ai/workflows/verify.md`.

## Branches, commits, and PRs

- Before work, inspect the current directory, `git status`, branch, and `git fetch` result. Preserve existing changes and stop on conflicts that need judgment.
- Never push directly to `main` or `develop`. Use a feature branch, focused commits, and a PR targeting `develop`. State dependencies on preceding PRs.
- Never force push or rewrite published history: no `--force`, `--force-with-lease`, `-f`, `+refspec` pushes, and no rebase of pushed branches. Update stacked branches with `git merge`. If a push is rejected as non-fast-forward, fetch and merge; stop if that needs judgment.
- Use `type(scope): Korean description`; allowed types are `feat`, `fix`, `docs`, `test`, `chore`, `refactor`. Write PR reasons, changes, verification, and related issues in Korean. Follow `docs/ai/workflows/commit-and-pr.md`.
- A PR is not done until review-bot comments are handled per `docs/ai/workflows/review-response.md`.
