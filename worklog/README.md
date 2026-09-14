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

4. **Stays local, not pushed to GitHub:** `.gitignore` excludes `worklog/*` (except this
   README and `_TEMPLATE.md`) — the actual per-branch worklog content is raw,
   in-progress research notes, not a polished public artifact, same reasoning as the
   `plans/` folder. It still lives on disk and any AI agent working in this repo can (and
   should) read/write it normally — "gitignored" only means "not synced to GitHub," not
   "off limits."

5. **Survives across branches, but not across clones:** since it's untracked, the file
   persists locally as long as the branch does, but won't show up if you clone the repo
   fresh elsewhere or if the branch folder gets deleted. If a worklog entry captures
   something worth permanently preserving (a real finding, a methodology decision),
   promote it into `MASTER_CARLA_RESEARCH_SUMMARY.md` or the PR description — don't rely
   on the worklog itself as permanent history.

## For AI agents

If you're an AI agent working in this repo — see `AGENTS.md` for the instruction to
maintain the current branch's worklog proactively as you work, not just when asked, and
for how to use it when asked to help write a weekly progress report.
