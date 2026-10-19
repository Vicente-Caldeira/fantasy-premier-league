.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check up down logs web-check tf-check api web

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install Python + web deps and git hooks
	uv sync
	cd apps/web && pnpm install
	uv run pre-commit install

lint: ## Lint Python
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format Python
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Type-check Python
	uv run mypy

test: ## Run Python tests with coverage
	uv run pytest --cov

web-check: ## Lint, type-check and build the web app
	cd apps/web && pnpm lint && pnpm typecheck && pnpm build

tf-check: ## Validate Terraform
	terraform fmt -check -recursive infra/terraform
	cd infra/terraform/bootstrap && terraform init -backend=false -input=false >/dev/null && terraform validate

check: lint typecheck test web-check tf-check ## Everything CI runs

api: ## Serve the API on :8000 (reads data/ and the warehouse)
	uv run uvicorn fpl_api.main:app --reload --reload-dir apps/api/src --port 8000

web: ## Serve the website on :3000 (needs `make api`)
	cd apps/web && pnpm dev

up: ## Start the full local stack
	docker compose up -d --build

down: ## Stop the local stack
	docker compose down

logs: ## Tail local stack logs
	docker compose logs -f
