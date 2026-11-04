# Commit and open a PR

1. Inspect the current directory, branch, working tree, and fetched remote changes. Confirm `develop` as the PR target; preserve unrelated changes.
2. Review the diff for secrets and scope. Run `docs/ai/workflows/verify.md`.
3. Make focused commits using `type(scope): Korean description`; allowed types are in `AGENTS.md`. Never push to `main` or `develop` directly.
4. Open a PR to `develop`. Fill `.github/pull_request_template.md` in Korean with reasons, changes, actual verification, related issues, and any preceding PR dependency.
5. Apply `docs/ai/workflows/review-response.md`. The PR is not done while actionable review-bot comments remain.
