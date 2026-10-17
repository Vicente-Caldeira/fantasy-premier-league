# FPL xPts

An end-to-end ML platform that predicts **Fantasy Premier League expected points** and turns them into transfer, captaincy and chip decisions. It is built to production standards on a $0/month cloud budget.

> Status: **Phase 0, foundations.** See the [roadmap](docs/architecture.md#roadmap).

## What it does

- Models every FPL scoring event per player and fixture: minutes, goals, assists, clean sheets, goals conceded, defensive contributions, saves and bonus. These are combined into expected points over a multi-gameweek horizon.
- Derives team goal expectations, and from them clean-sheet probabilities, from betting market odds.
- Optimises squads, transfers, captaincy and chip timing with mixed-integer programming.
- Hosted web app with Google sign-in that remembers your FPL team.

## Architecture

```
Sources (FPL API, FPL-Elo-Insights, Understat, odds)
  → GCS bronze → BigQuery + dbt → Feast → training (MLflow) → backtest gate
  → batch inference → Postgres → FastAPI → Next.js
                     orchestrated by Dagster · infra by Terraform · CI/CD by GitHub Actions
```

- [Architecture & stack](docs/architecture.md)
- [FPL rules & data sources research](docs/research.md)
- [Architecture decision records](docs/adr/)

## Repository layout

| Path | Contents |
|---|---|
| `packages/fpl_core` | Domain model: positions, scoring rules, shared schemas |
| `packages/fpl_models` | Features, component models, backtesting, optimiser |
| `pipelines/` | Dagster code location (ingest, transform, train, infer) |
| `apps/api` | FastAPI service |
| `apps/web` | Next.js frontend |
| `infra/terraform` | GCP infrastructure, including the billing kill-switch |
| `docker/` | Container images and local stack config |

## Local development

Prerequisites:
- [uv](https://docs.astral.sh/uv/)
- Node 26 + pnpm
- Docker (with buildx)
- Terraform

```sh
make install   # Python + web deps, git hooks
make check     # everything CI runs
make up        # full local stack
```

| Service | URL |
|---|---|
| Web | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Dagster | http://localhost:3001 |
| MLflow | http://localhost:5050 |

## License

[MIT](LICENSE)
