# 3. Cloud Run instead of Kubernetes

- Status: accepted
- Date: 2026-10-17

## Context

Kubernetes is common in big tech, but GKE has a per-cluster management fee and always-on nodes, which breaks the $0 constraint (ADR 0002). The workloads are an HTTP API, a web frontend, and batch jobs that run a few times a day.

## Decision

- Run services on Cloud Run services and batch work on Cloud Run Jobs.
- Ship everything as OCI images built from Dockerfiles in `docker/`, with no Cloud Run-specific code. Configuration comes from environment variables, the port from `$PORT`, and logs go to stdout.

## Consequences

- No cluster to operate, and it scales to zero.
- Moving to Kubernetes later only needs manifests or Helm charts; the images stay the same.
- Constraints we accept: jobs are limited to Cloud Run's maximum task timeout, and there is no sidecar-heavy service mesh.
