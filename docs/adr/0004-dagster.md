# 4. Dagster for orchestration

- Status: accepted; hosted scheduling superseded by [ADR 0006](0006-spend-caps-and-free-compute.md)
- Date: 2026-10-17

## Context

Pipelines are naturally modelled as data assets, such as raw snapshots, dbt models, features, models and predictions. They are partitioned by season and gameweek and need backfills. Airflow is the most widely known option, but managed Airflow (Composer) isn't free, and Airflow's task-centric model is a poorer fit for asset lineage.

## Decision

- Use Dagster OSS, with a code location in `pipelines/` and Postgres run storage.
- Locally, docker-compose runs the webserver and daemon.
- When hosted:
  - Cloud Scheduler triggers Cloud Run Jobs that run `dagster job execute` against the same Postgres (Neon) storage.
  - The webserver runs on Cloud Run, scaled to zero, for lineage and history.
  - No always-on daemon.

## Consequences

- Asset lineage, partitions, and integration with dbt and MLflow come built in.
- Without a hosted daemon, Dagster sensors and schedules don't run in production. Scheduling lives in Cloud Scheduler, which Terraform defines.
