# Respond to PR reviews

1. For each PR, query `gh api repos/9-jodae/docs199-AI/pulls/{n}/reviews`, `/pulls/{n}/comments`, and `/issues/{n}/comments`. If there is no review, poll every minute for up to ten minutes. Identify bots by account type `Bot` or a `[bot]` suffix. Stop on a human `CHANGES_REQUESTED` review.
2. Classify each new bot finding. **Apply** bugs, security issues, rule violations, and missing tests on that PR branch. Run `ruff check . && pytest`, commit with `fix(review): Korean description`, push, and reply with the change and commit SHA. **Disagree** only when the finding conflicts with `AGENTS.md`, the Korean `docs/`, or confirmed design, or is a false positive; reply with evidence and make no code change. **Defer** valid out-of-scope work to a new Issue and reply with its number.
3. Handle one follow-up review after a fix, for at most two rounds total. Reply to a repeated finding once. Stop if an actionable finding remains after round two.
4. Inspect review thread state with `gh api graphql -f query='...'` using the query below. Page through `reviewThreads` and nested `comments` if `hasNextPage` is true. After replying, resolve a thread only when its first comment came from a bot (cross-check the REST author's `type` or `[bot]` login), no human review request is pending, and no actionable finding remains. A human reply to a bot finding does not change thread ownership. Call `gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' -F id=THREAD_ID`. Check that `isResolved` is true. Never resolve a thread started by a human.

   ```graphql
   query {
     repository(owner: "9-jodae", name: "docs199-AI") {
       pullRequest(number: PR_NUMBER) {
         reviewThreads(first: 100, after: CURSOR_OR_NULL) {
           nodes {
             id
             isResolved
             comments(first: 100) {
               nodes { author { login } }
               pageInfo { hasNextPage endCursor }
             }
           }
           pageInfo { hasNextPage endCursor }
         }
       }
     }
   }
   ```
5. For stacked PRs, if a preceding branch changes, run `git merge origin/<preceding-branch>` on each later branch. If it diverges after an earlier PR merges, run `git merge origin/dev`. Stop if a conflict requires judgment. Verify, then use a normal push. Before merging, check passing CI, no actionable findings, no human `CHANGES_REQUESTED`, `mergeable` without conflicts, and base `dev`. Merge in order with `gh pr merge {n} --merge`; never use `--admin`, squash, rebase merge, or delete the remote branch. Recheck later PRs after each merge.
6. Report received findings, applied/disagreed/deferred counts, fix SHAs, Issue numbers, CI outcomes, and merge SHAs in Korean. For a review-only PR, do not merge it.
