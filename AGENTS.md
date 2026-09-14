# Project instructions for AI agents

This is Quentin's senior-year (Fall 2026) CS Independent Research project at Connecticut
College: safety-focused reinforcement learning for autonomous driving in CARLA, advised
by Prof. Ozgur Izmirli. Read `MASTER_CARLA_RESEARCH_SUMMARY.md` for full research context
and `CONTRIBUTING.md` for git branching/commit/PR conventions before making changes.

Quentin works across multiple AI tools/models (Claude and ChatGPT/Codex, inside Cursor
and elsewhere) — this file is the one canonical, tool-agnostic source of instructions so
they don't drift out of sync between tools. `CLAUDE.md` and `.cursor/rules/worklog.mdc`
both just point back here.

## Worklogs — read before doing development work on a branch

Every branch other than `develop` has a running worklog at `worklog/<branch-name>.md`,
auto-created by a git hook on branch checkout (see `worklog/README.md`). This is
Quentin's primary source material for mandatory weekly research progress reports to his
advisor — treat maintaining it as part of the task, not optional overhead. It stays local
(`.gitignore`d) rather than pushed to GitHub — see `worklog/README.md` for why — but it's
still a real file on disk; read and write it normally.

**Whenever a meaningful change happens while working on a branch** — something gets
implemented, an experiment runs, a bug is hit, something is tried and fails, a result
comes back — append a dated entry to that branch's worklog file, in the format described
in `worklog/_TEMPLATE.md`. Do this as work happens, not only when asked, and not only at
the very end of a session (if the session gets interrupted, the log should still reflect
what happened up to that point).

A good entry captures: what changed and why, what was tried, what worked, what didn't
(and why, if known), and any open questions or blockers for next time. Skip entries for
trivial/mechanical changes (typo fixes, formatting) — this is a research log, not a
commit log.

**When asked to help write a weekly progress report:** read the worklog file(s) for
whichever branch(es) had activity that week (check `worklog/` for recently-modified
files if unsure which), and summarize the entries into Quentin's tracker format
(Week / Goal / Time Log / Progress) — this should be compression of what's already
recorded, not a request to reconstruct the week from conversation history, since a
different AI tool may have done some of the actual work.

When a PR is opened for the branch, draw on the worklog to fill in
`.github/PULL_REQUEST_TEMPLATE.md` — the Changes / Testing / Results sections should
largely be a distillation of worklog entries, not written from scratch.

## Everything else

Follow `CONTRIBUTING.md` for branch naming, commit message format, and the pre-push
checklist. Don't push large binaries (checkpoints, raw run data, raw camera captures) —
see `.gitignore` for what's already excluded and why.
