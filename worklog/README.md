# Worklogs

A worklog is a running, dated record of what happened on one branch — what changed, why,
what worked, what didn't. It exists so that writing the weekly advisor progress report,
or a PR description, is compression of existing notes instead of trying to reconstruct a
week of work from memory or `git log`.

## How it works

1. **One-time per clone:** enable the repo's tracked git hooks —
   ```
   git config core.hooksPath .githooks
   ```
   (Already covered in the main setup steps in `CONTRIBUTING.md` / `README.md`.)

2. **Automatic:** the first time you check into a new branch (`git checkout -b
   experiment/reward-shaping-v4`), a `post-checkout` hook auto-creates
   `worklog/experiment/reward-shaping-v4.md` from a template — nested to match the branch
   name. `develop` itself never gets one; worklogs are for in-progress work.

3. **Manual:** you (or an AI agent helping you) append a dated entry to that file
   whenever something meaningful happens — see `_TEMPLATE.md` for the exact format and an
   example.

4. **It ships with the branch:** commit the worklog file alongside your actual code
   changes, and it travels with the PR — a reviewer (or future you) gets the process
   record right next to the diff, not hunting through chat history to reconstruct it.

5. **After merge:** the worklog stays in `develop`'s history permanently (it's a real
   part of the research record, same spirit as `MASTER_CARLA_RESEARCH_SUMMARY.md`) — it
   is not meant to be deleted once its branch is merged.

## For AI agents

If you're an AI agent working in this repo — see `CLAUDE.md` (or, in Cursor, the rule in
`.cursor/rules/`) for the instruction to maintain the current branch's worklog
proactively as you work, not just when asked.
