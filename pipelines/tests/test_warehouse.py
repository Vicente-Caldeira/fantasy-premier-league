"""dbt manifest freshness: re-parse whenever a project file changed after the last parse."""

import os

from fpl_pipelines.assets.warehouse import manifest_is_stale


def test_manifest_is_stale_when_missing_or_older_than_a_model(tmp_path):
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "fct_x.sql"
    model.write_text("select 1")
    (tmp_path / "dbt_project.yml").write_text("name: fpl")
    manifest = tmp_path / "target" / "manifest.json"
    assert manifest_is_stale(manifest, tmp_path)
    manifest.parent.mkdir()
    manifest.write_text("{}")
    os.utime(model, (1_000, 1_000))
    os.utime(tmp_path / "dbt_project.yml", (1_000, 1_000))
    assert not manifest_is_stale(manifest, tmp_path)
    os.utime(model, None)  # edited after the parse
    os.utime(manifest, (2_000, 2_000))
    assert manifest_is_stale(manifest, tmp_path)
