# Worklog: chore/add-worklog-system

Created: 2026-09-14 14:01

> Append a dated entry every time a meaningful change happens on this branch — human or
> AI agent. Entry format: worklog/_TEMPLATE.md. This is the raw material for Quentin's
> weekly advisor progress report and for the PR description when this branch is ready.
> See worklog/README.md for the full system.

## Goal

Build a system so that work on any branch (feature/experiment/fix/etc.) automatically
gets a running, dated worklog — capturing what changed, why, what worked/didn't — so
weekly advisor progress reports and PR descriptions can be written from existing notes
instead of reconstructed from memory at the end of the week.

## Log

### 2026-09-14 14:00

**What:** Added a `post-checkout` git hook (`.githooks/post-checkout`) that
auto-scaffolds `worklog/<branch-name>.md` the first time a branch is checked into
(skipping `develop`). Added `worklog/README.md` (system explanation) and
`worklog/_TEMPLATE.md` (entry format + example). Added `CLAUDE.md` and
`.cursor/rules/worklog.mdc` so AI agents in either Claude Code or Cursor know to
proactively maintain the current branch's worklog. Updated `CONTRIBUTING.md` and
`README.md` with the one-time `git config core.hooksPath .githooks` setup step.
**Why:** Requested — wanted an automatic (not remember-to-do-it) way to keep a record of
in-progress research work, usable both to inform AI agents helping with development and
to compile weekly progress reports for the research advisor.
**Result:** Worked. Verified the hook fires correctly by creating a throwaway
`test/hook-verification` branch and confirming `worklog/test/hook-verification.md` was
created with the right content, then deleted the test branch.
**Next:** None outstanding — ready for PR. Worth revisiting after a few weeks of real use
to see if the entry format needs adjusting (e.g. if entries end up too sparse or too
verbose to actually compress into a weekly report easily).

