# Research artifacts

This directory contains generated evidence and trained-model artifacts. The active
implementation lives in `src/`; automated tests live in `tests/`; historical source
snapshots live in `previous_src_tests/`.

## Layout

- `runs/` contains experiment outputs, retained summaries, plots, manifests, logs,
  and (where preserved locally) raw per-run traces.
- `checkpoints/` contains trained SAC models and intermediate training checkpoints.

Most large/raw contents are intentionally git-ignored. Compact, historically important
summaries and the small set of already-versioned model-lineage ZIPs remain tracked.

## Guidance for people and AI agents

Do not recursively read or summarize this entire directory for ordinary code changes.
Start with source, tests, `README.md`, and the relevant section of
`MASTER_CARLA_RESEARCH_SUMMARY.md`.

Artifacts are evidence, not forbidden material. Inspect the smallest relevant subset
when a task concerns experimental results, regression evidence, model provenance,
reproducibility, or a historical finding. Prefer, in order:

1. manifests and summary CSV/JSON files;
2. named reports and progress logs;
3. individual run metrics;
4. raw traces only when the question requires tick-level evidence.

Never delete, overwrite, or bulk-regenerate retained artifacts without first confirming
their provenance and whether another copy exists.
