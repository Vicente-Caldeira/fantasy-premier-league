"""dbt models as Dagster assets, downstream of the silver tables they read.

Sources declare `meta.dagster.asset_key`, so each dbt model is wired to the ingestion
asset that produces its inputs. The manifest is built with `dbt parse`, which needs no
data, so definitions load anywhere (CI, containers, tests).
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dagster as dg
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

DBT_DIR = Path(os.getenv("FPL_DBT_DIR", Path(__file__).parents[4] / "dbt"))

dbt_project = DbtProject(project_dir=DBT_DIR, profiles_dir=DBT_DIR)


def manifest_is_stale(manifest: Path, project_dir: Path = DBT_DIR) -> bool:
    """Missing, or older than any dbt project file (models, tests, seeds, macros, config):
    a stale manifest makes Dagster's asset graph disagree with what dbt runs."""
    if not manifest.exists():
        return True
    built = manifest.stat().st_mtime
    sources = [project_dir / "dbt_project.yml"]
    for folder in ("models", "tests", "seeds", "macros", "snapshots"):
        sources.extend((project_dir / folder).rglob("*"))
    return any(f.is_file() and f.stat().st_mtime > built for f in sources)


def ensure_manifest() -> Path:
    if manifest_is_stale(Path(dbt_project.manifest_path)):
        # dagster-dbt gives each invocation its own target dir; parse into the canonical one.
        DbtCliResource(project_dir=dbt_project).cli(
            ["parse", "--quiet"], target_path=Path(dbt_project.target_path)
        ).wait()
    return Path(dbt_project.manifest_path)


@dbt_assets(manifest=ensure_manifest(), name="warehouse")
def warehouse(context: dg.AssetExecutionContext, dbt: DbtCliResource) -> Iterator[Any]:
    """Build every dbt model and run its tests; a failing test fails the run."""
    yield from dbt.cli(["build"], context=context).stream()
