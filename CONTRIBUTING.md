# Working conventions

This is a solo research repo, but it follows the same conventions a small professional team
would use — mostly so the history stays readable to *future you* (and to anyone, like an
advisor, who looks at it). This doc is the reference for how to branch, commit, and open PRs.

## Branching

Everything happens on a branch off `develop`; `develop` itself only receives merges via PR
(see [When to PR vs. push directly](#when-to-pr-vs-push-directly) for the one exception).

**Naming:** `<type>/<short-kebab-case-description>`

| Type | Use for |
|---|---|
| `feature/` | New capability (a new scenario type, a new metric, a new script) |
| `experiment/` | A training run / hyperparameter sweep / research question you're testing — not guaranteed to "work" |
| `fix/` | Correcting a bug in existing code |
| `refactor/` | Restructuring code with no behavior change |
| `docs/` | Documentation only (including updates to `MASTER_CARLA_RESEARCH_SUMMARY.md`) |
| `chore/` | Tooling, `.gitignore`, dependency, or workflow changes |

Examples: `experiment/reward-shaping-v4`, `fix/brake-calibration-offset`, `docs/update-repo-guide`.

**Nested / stacked branches:** if you need to start new work on top of a branch that isn't
merged yet (e.g. an experiment that depends on an in-progress refactor), branch off that branch
instead of `develop`, and name it `<parent-type>/<parent-topic>--<sub-topic>`
(e.g. `refactor/env-cleanup--add-lidar-noise`). Say in the PR description what it's actually
based on, since GitHub's compare view will otherwise diff against `develop` and show unrelated
changes. Merge (or rebase) the parent first when possible — keep the stack shallow (one level
deep) rather than chaining several branches on top of each other.

## Commit messages

Format (loosely [Conventional Commits](https://www.conventionalcommits.org/)):

```
<type>: <imperative summary, ~50 chars, no trailing period>

<optional body: why this change, not what it does line-by-line —
the diff already shows what changed>
```

Use the same `<type>` prefixes as the branch table above (`feat`, `fix`, `experiment`, `refactor`,
`docs`, `chore`). Write the summary as an instruction ("add reward penalty for late braking"),
not a description ("added reward penalty").

A one-time local setup makes this the default when you type `git commit` (no `-m`):

```
git config commit.template .gitmessage
```

## Pull requests

**Title:** same convention as commit messages — `<type>: <summary>`.

**Description:** filled in from `.github/PULL_REQUEST_TEMPLATE.md` automatically when you open
the PR on GitHub. Don't leave sections blank — write "N/A" if genuinely not applicable, so it's
clear you considered it rather than skipped it.

## Pre-push checklist

Run through this before pushing a branch, and again before merging the PR:

- [ ] **It runs.** Actually execute the script(s) you changed (or the relevant CARLA scenario)
      end-to-end at least once — a training loop that only fails after 500k steps is much cheaper
      to catch by running it briefly than by discovering it in the middle of an overnight run.
- [ ] **No large or generated files snuck in.** Run `git status` before committing and check the
      file list against `.gitignore` — checkpoints, `runs/*/run_*/` folders, and camera captures
      should never appear staged. If one does, fix `.gitignore` rather than committing it once
      "just this time."
- [ ] **No dead/commented-out experiment code left in** unless it's genuinely useful reference —
      delete it instead (git history keeps it if you need it back).
- [ ] **Docs updated if behavior or results changed** — `MASTER_CARLA_RESEARCH_SUMMARY.md` is
      described as "the authoritative starting point," so if a result in it becomes stale because
      of this change, update it (or note the discrepancy) in the same PR.
- [ ] **Self-review the diff** on GitHub before merging (`Files changed` tab on the PR) — reading
      your own diff in that view catches things a local `git diff` doesn't, like accidental
      whitespace changes or leftover debug prints.

## When to PR vs. push directly

Open a PR (even to yourself, even if you merge it five minutes later) for anything touching
code, configs, or results — it creates a reviewable, timestamped record of *why* a change
happened, which matters for a research project you'll need to explain later (in a paper, or to
an advisor). Pushing straight to `develop` is fine only for trivial, zero-risk edits: fixing a
typo in a doc, or updating this file.
