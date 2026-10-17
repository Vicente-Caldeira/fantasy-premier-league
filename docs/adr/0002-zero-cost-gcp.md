# 2. Host on GCP Always Free with a billing kill-switch

- Status: accepted
- Date: 2026-10-17

## Context

The hosted app must cost $0/month. It should still use a hyperscaler, so that the infrastructure resembles what big tech teams run. GCP's free tier requires a billing account on file.

## Decision

- Run on GCP in `us-central1`, using only Always Free services: Cloud Run, Cloud Scheduler, BigQuery, GCS, Artifact Registry, Pub/Sub and Firebase Auth.
- Replace paid-only services with free managed equivalents: Cloud SQL becomes Neon Postgres.
- Add a kill-switch. A $1 budget publishes to Pub/Sub, which triggers a Cloud Run function that unlinks the project's billing account once actual spend exceeds the budget.
- Terraform validation rejects regions that aren't in the free tier.

## Consequences

- Budget alerts lag real spend by hours, so the kill-switch is a backstop. The real defence is to avoid always-on or non-free resources: no min-instances, VPC connectors, NAT or load balancers.
- Disabling billing stops every paid service in the project, which takes the app down rather than running up cost. That is the intended trade-off.
- Cold starts on scale-to-zero services are accepted.
