# 5. Layered spend guardrails

- Status: accepted
- Date: 2026-10-17
- Extends: [ADR 0002](0002-zero-cost-gcp.md)

## Context

GCP has no hard spending cap. Budgets are evaluated hours after usage. The billing account is in EUR and may hold trial credits, which budgets subtract by default and which would therefore hide real spend.

## Decision

The layers below are ordered by how directly each one stops spend:

1. **Service quotas** (enforced in real time):
   - BigQuery query usage is capped at 30 GiB/day per project, so it can't exceed the 1 TiB/month free tier.
   - Cloud Run services, jobs and functions always set `max_instance_count`.
2. **Project kill-switch:** a €1 budget publishes to Pub/Sub, and a function unlinks billing from the project.
3. **Account-wide budget:** €1 across every project on the billing account. It emails billing admins at 1%, 50% and 100%.
4. Both budgets use `INCLUDE_SPECIFIED_CREDITS` with only `FREE_TIER` subtracted, so trial or promotional credits never mask spend.
5. **Outside GCP** (manual and optional): pay with a virtual card that has a low limit. A declined charge suspends the billing account.

## Consequences

- Any new service we adopt must come with a quota or instance cap before it ships.
- A Cloud Run free-tier credit is not visible until billing data lands, so small transient spend may appear in budget emails before it is netted off.
