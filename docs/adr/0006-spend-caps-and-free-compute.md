# 6. Spend caps, free batch compute and cost conventions

- Status: accepted
- Date: 2026-10-17
- Extends: [ADR 0005](0005-spend-guardrails.md). Supersedes the hosted scheduling in [ADR 0004](0004-dagster.md)

## Context

A review of the guardrails against Google's documentation found the following:

- **Spend caps exist.** Google offers *spend cap budgets*, which pause a service once its cost reaches the cap. They cover Cloud Run and Cloud Run functions only, not BigQuery, Storage or Artifact Registry. They count **gross** cost and enforcement lags, so overage from that lag is still billed.
- **Cloud Run Jobs may not be free.** The documented Cloud Run free tier covers *request-based billing*. We couldn't confirm it covers Cloud Run Jobs.
- **Image storage could exceed the free tier.** Cloud Functions builds store images in an auto-created `gcf-artifacts` repo with no cleanup, which could grow past the 0.5 GB Artifact Registry free tier.
- **Default service accounts are a risk.** The kill-switch relied on default compute service account grants that newer projects may not have.
- **The kill-switch was never tested.** Nothing proved it would work when needed.

## Decision

1. **Spend caps:**
   - Cloud Run is capped at €2/month gross and Cloud Run functions at €1/month gross.
   - The Cloud Run free tier is worth about €5/month gross, so even a full cap plus lag stays inside it and nets to €0.
   - The provider doesn't support spend caps yet, so they're created through the Budget API v1 (`scripts/spend_cap.py`, called from `terraform_data`).
2. **Batch compute:** pipelines run on scheduled GitHub Actions workflows, which are free and unlimited for public repos. Cloud Run hosts only request-billed services.
3. **Kill-switch hardening:**
   - It has dedicated build, trigger and runtime service accounts.
   - Its images go to a repository with a cleanup policy that keeps only the latest one.
   - Malformed messages are dropped, while a failed unlink raises so Pub/Sub retries it.
   - A `dry_run=true` message runs a self-test that checks the unlink permission without unlinking anything. This is run after every apply.
4. **Cost conventions** for all future work:
   - BigQuery loads use load jobs, never legacy streaming inserts, which have no free tier.
   - At most 6 active Secret Manager versions.
   - Every Cloud Run service sets `max_instance_count`.
   - App pages require sign-in, and the API rate-limits requests, which bounds network egress (1 GB/month free, and not covered by spend caps).
   - All storage stays in `us-central1`.

## Consequences

- Any accidental spend is hard-capped for the services where caps exist. Quota caps, budgets and the kill-switch cover the rest.
- GitHub Actions minutes become the batch compute budget, with a 6-hour job limit.
- Remaining exposure, which can't be capped:
  - network egress beyond 1 GB/month;
  - storage beyond the free tiers.

  Both are bounded by design and caught by the €1 budgets.
