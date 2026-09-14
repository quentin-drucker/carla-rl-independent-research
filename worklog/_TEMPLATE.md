# Worklog entry format

Append entries to the bottom of your branch's `worklog/<branch-name>.md` under `## Log`,
newest at the bottom (chronological, matches how the work actually happened). One entry
per meaningful chunk of work — not per commit, not per keystroke. Skip trivial/mechanical
changes (typo fixes, formatting); this is a research log, not a commit log (git already
has that).

```
### YYYY-MM-DD HH:MM

**What:** What changed, or what was attempted.
**Why:** What question, goal, or problem this addresses.
**Result:** worked / didn't work / partial / inconclusive — and why, if known.
**Next:** Open questions, blockers, or what to try next.
```

## Example

```
### 2026-09-21 14:30

**What:** Added a second continuous action (steer) to CarlaAEBEnv's action space;
retrained SAC for 50k steps as a smoke test (not a real training run).
**Why:** First step toward the steering pivot — wanted to confirm the env/training loop
still runs end-to-end with a 2D action space before committing to a real run.
**Result:** Worked — training completes, but reward is unstable after ~30k steps.
Suspect the reward function still assumes brake-only behavior (no penalty for
unnecessary steering).
**Next:** Reward function needs a steering term before a real training run is worth the
compute. Also need to decide whether avoidability.py's straight-line physics model can
be reused or needs a lateral-motion version.
```

## Using this at report/PR time

- **Weekly progress report:** skim the week's entries across your active branch(es) and
  summarize into your tracker's Goal/Progress columns — the entries already have the
  what/why/result, so this should mostly be compression, not re-remembering.
- **PR description:** `.github/PULL_REQUEST_TEMPLATE.md`'s Changes / Testing / Results
  sections should draw directly from the relevant entries, not be written from scratch.
