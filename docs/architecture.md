# FPL App: Architecture & Tech Stack

_v0.2, 2026-10-17. Decisions are recorded as ADRs in `docs/adr/`._

**Hard constraint: $0/month.** Everything runs inside GCP's Always Free tier plus free third-party tiers, with a billing kill-switch as a backstop ([ADR 0002](adr/0002-zero-cost-gcp.md)).

## Goals

1. **Product:** a hosted web app with Google sign-in that remembers your FPL team. It shows per-player expected points over the next N gameweeks, and gives transfer, captaincy and chip recommendations.
2. **Engineering:** the system should look and behave like a production ML platform. That means:
   - reproducible data pipelines;
   - point-in-time-correct features;
   - tracked experiments and a model registry;
   - automated backtesting gates;
   - monitored, versioned predictions;
   - infrastructure as code;
   - CI/CD.

Non-goals for v1: mobile apps, social features, paid tiers.

## System overview

```
            ┌────────────── Ingestion (Dagster assets, scheduled + sensors) ──────────────┐
 FPL API ───┤                                                                            │
 FPL-Elo ───┤──► Bronze: raw JSON/CSV in GCS (immutable, partitioned by date/GW) ────────┤
 Understat ─┤                                                                            │
 Odds (fd.co.uk, Odds API) ┘                                                            │
                                                                                         ▼
                     Silver/Gold: BigQuery, modelled with dbt (tests + docs + lineage)
                                                                                         │
                         Feast feature store (offline: BigQuery, online: Postgres)
                                                                                         │
          Training (Dagster job) ──► MLflow (experiments, registry) ──► Backtest gate
                                                                                         │
          Batch inference per GW ──► predictions table (versioned by model + data snapshot)
                                                                                         │
     FastAPI service (predictions, optimiser, user team) ◄── Postgres (users, teams, preds)
                                                                                         │
                     Next.js web app (Google sign-in via Firebase Auth)
                                                                                         │
            Monitoring: Evidently (drift + realised-vs-predicted), OpenTelemetry, Sentry
```

## Stack decisions

| Area | Choice | Why |
|---|---|---|
| Cloud | **GCP, Always Free tier in `us-central1`**, plus a budget → Pub/Sub → function kill-switch that unlinks billing above $1 | Native Google identity, BigQuery free tier, and Cloud Run scales to zero. All of it maps directly to AWS/Azure equivalents |
| IaC | **Terraform** (separate `dev` and `prod` workspaces) | Industry standard; reviewable infra changes |
| Containers / compute | **Docker + Cloud Run** (services + jobs) | Production-grade serverless that scales to zero. Images stay K8s-portable. GKE is skipped because of cost ([ADR 0003](adr/0003-cloud-run-over-gke.md)) |
| Python tooling | **Python 3.12, uv, Ruff, mypy (strict), pytest, Hypothesis, pre-commit** | Modern, fast, typed |
| Orchestration | **Dagster**. Locally: webserver + daemon. Hosted: scheduled **GitHub Actions** workflows run Dagster jobs (free and unlimited for public repos), and run storage lives in Neon | Asset-based lineage, partitions per gameweek, and a first-class dbt integration. No paid compute for batch work ([ADR 0004](adr/0004-dagster.md), [ADR 0006](adr/0006-spend-caps-and-free-compute.md)) |
| Raw storage | **GCS** (bronze, immutable) | Lets us replay and backfill from the raw source data |
| Warehouse + transforms | **BigQuery + dbt** | Standard analytics engineering, with tests, docs and lineage |
| Data contracts | **Pydantic** (API payloads), **Pandera** (dataframes), **dbt tests** | Bad upstream data fails loudly |
| Feature store | **Feast** | Teaches point-in-time joins, which stop future data leaking into training. Also gives offline and online parity |
| Modelling | **statsmodels / PyMC** (team-strength Poisson / Dixon–Coles, hierarchical player shares), **LightGBM** (minutes, DefCon, bonus), **scikit-learn** calibration | Interpretable component models per scoring event, combined into expected points |
| Experiment tracking + registry | **MLflow** (Cloud Run scaled to zero, Neon Postgres backend, GCS artefacts) | Portable and widely recognised |
| Evaluation | Walk-forward backtests per GW over several seasons: MAE/RMSE on points, Brier score / log loss on CS and DefCon probabilities, calibration plots, and baselines (FPL `ep_next`, form) | A model is promoted only if it beats the current champion. CI enforces this |
| Optimisation | **OR-Tools** (MILP): squad selection, multi-GW transfer planning, captaincy, chip timing | Proper constrained optimisation |
| Backend API | **FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic** | Typed and async, with an OpenAPI schema that generates the frontend client |
| App DB | **Neon serverless Postgres (free tier)**, with separate databases for app, Dagster and MLflow | Cloud SQL has no free tier. Holds users, linked FPL entry IDs, cached predictions, and the Feast online store |
| Auth | **Firebase Authentication (Google provider)** | The frontend gets an ID token and FastAPI verifies the JWT. User → FPL team ID is stored in Postgres |
| Frontend | **Next.js (TypeScript), Tailwind, shadcn/ui, TanStack Query**, generated OpenAPI client | Standard modern React stack |
| CI/CD | **GitHub Actions** + Workload Identity Federation (no service-account keys), Artifact Registry, preview envs | Lint → type-check → test → build → deploy dev → gated deploy prod |
| Observability | **OpenTelemetry → Cloud Monitoring/Trace, Sentry (free tier), Evidently** | Covers service health, plus data and model drift. A weekly "how wrong were we" report |
| Secrets | **Secret Manager** | |
| Docs | Design doc, ADRs, model cards, data dictionary (dbt docs), runbooks | Shows engineering maturity to reviewers |

## Repo layout (monorepo)

```
fantasy-premier-league/
├── apps/
│   ├── api/                 # FastAPI service
│   └── web/                 # Next.js frontend
├── pipelines/               # Dagster code location (ingestion, training, inference)
├── packages/
│   ├── fpl_core/            # shared domain: scoring rules, schemas, clients
│   └── fpl_models/          # model code, features, backtesting, optimiser
├── dbt/                     # warehouse models
├── feature_repo/            # Feast definitions
├── infra/terraform/         # GCP infra (modules + envs/dev, envs/prod)
├── docs/                    # design doc, ADRs, model cards, runbooks
├── .github/workflows/
└── docker-compose.yml       # full local stack (Postgres, MLflow, Dagster, API, web)
```

## Roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0. Foundations | Repo skeleton, tooling, CI, docker-compose, Terraform bootstrap | CI is green and the local stack comes up with one command |
| 1. Data platform | Bronze ingestion for all sources, BigQuery + dbt silver/gold, data tests, historical backfill (2016→now) | Every gameweek is reproducible from raw data, and the dbt docs are published |
| 2. Baseline + eval harness | Walk-forward backtester, baselines (`ep_next`, form, PPG) | We have a leaderboard to beat |
| 3. Component models | Team λ (odds + strength), minutes, goal/assist shares, CS, DefCon, saves, bonus → xPts | Beats the baselines on the backtest, and the probabilities are calibrated |
| 4. Serving | Batch inference per GW, FastAPI, Postgres, MLflow registry promotion gate | Predictions are versioned, and the API meets its latency SLO |
| 5. Product | Next.js app, Google sign-in, team import, xPts tables, fixture ticker | Deployed on prod, behind auth |
| 6. Optimiser | Transfers, captain, chips (OR-Tools) | Recommendations are explainable |
| 7. MLOps hardening | Drift monitoring, weekly performance report, alerting, retraining triggers | A runbook for each alert |

## Running cost: $0

| Resource | Free allowance | Our expected usage |
|---|---|---|
| Cloud Run | 2M requests, 180k vCPU-s/month | Low traffic. Services scale to zero |
| Cloud Run spend caps | Google pauses Cloud Run / functions at €2 / €1 **gross** cost | Well below the free-tier value |
| BigQuery | 10 GB storage, 1 TB queries/month | Under 5 GB total |
| GCS (`us-central1`) | 5 GB | Raw snapshots, compressed |
| Artifact Registry | 0.5 GB | Slim images + cleanup policy keeping the last 2 tags |
| Firebase Auth | Free for Google sign-in | |
| Neon Postgres | 0.5 GB per project | App + Dagster + MLflow metadata |
| GitHub Actions | Unlimited for public repos | CI, plus a fallback scheduler |

**Guards:**
- Free-tier regions are enforced in Terraform.
- No VPC connectors, NAT, load balancers or min-instances (none of these have a free tier).
- The kill-switch unlinks billing once spend exceeds €1. See [ADR 0005](adr/0005-spend-guardrails.md) and [ADR 0006](adr/0006-spend-caps-and-free-compute.md) for the full set of guardrails.

## Notes

- **FPL API:** it is public but undocumented. Responses get cached, we use one polite scheduled poller with backoff, and end users never hit it directly.
- **Source data:** we show derived insights rather than republishing it wholesale.
