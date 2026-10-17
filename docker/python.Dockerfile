# Shared image for the Python workspace members. Build with:
#   docker build -f docker/python.Dockerfile --build-arg PACKAGE=fpl-api .
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
ARG PACKAGE
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Resolve third-party deps first so they cache independently of source changes.
COPY pyproject.toml uv.lock ./
COPY packages/fpl_core/pyproject.toml packages/fpl_core/
COPY packages/fpl_models/pyproject.toml packages/fpl_models/
COPY apps/api/pyproject.toml apps/api/
COPY pipelines/pyproject.toml pipelines/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package "${PACKAGE}"

COPY packages packages
COPY apps/api apps/api
COPY pipelines pipelines
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package "${PACKAGE}"

FROM python:${PYTHON_VERSION}-slim AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app dbt /app/dbt
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    FPL_DBT_DIR=/app/dbt
WORKDIR /app
USER app
# Bake the dbt manifest into the image so containers never parse on start-up.
RUN cd /app/dbt && dbt parse --profiles-dir . --quiet
