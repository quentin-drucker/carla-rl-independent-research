# CARLA RL Independent Research

Senior-year (Fall 2026) CS Independent Study at Connecticut College, continuing safety-focused
reinforcement learning research in the [CARLA](https://carla.org/) simulator — training an SAC
agent to perform autonomous emergency braking (AEB) across parameterized rare-hazard scenarios.
This work began as a research project for COM496 (junior spring, 2026).

## Where to start

- **[MASTER_CARLA_RESEARCH_SUMMARY.md](MASTER_CARLA_RESEARCH_SUMMARY.md)** — the authoritative
  project reconstruction: background, architecture, repository guide, how to run things, results,
  and open questions. Start here.
- **[WORKFLOW.md](WORKFLOW.md)** — day-to-day working notes.
- **[notes/](notes)** — earlier research notes from the original semester.

## Environment setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires a running CARLA 0.9.16 simulator instance (see the setup section in
`MASTER_CARLA_RESEARCH_SUMMARY.md` for details).

## Repository layout

- `src/` — current codebase: environment, training, evaluation, and plotting scripts.
- `previous_src_tests/` — earlier exploratory test scripts, kept for reference.
- Large binaries (trained model checkpoints, raw per-run telemetry, raw camera captures) are
  intentionally excluded from version control — see `.gitignore`.

## Branching

Ongoing work happens on branches off `develop`; changes land back in `develop` via pull request
once ready.
