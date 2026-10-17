"""Contracts validated against trimmed, real API payloads captured on 2026-10-17."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_core.fpl_api import BootstrapStatic, ElementSummary, Fixture

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_bootstrap_static_parses_real_payload():
    bootstrap = BootstrapStatic.model_validate(load("bootstrap_static.json"))
    assert bootstrap.season == "2026-27"
    assert [p.singular_name_short for p in bootstrap.element_types] == ["GKP", "DEF", "MID", "FWD"]
    assert bootstrap.elements


def test_string_decimals_are_coerced_to_floats():
    player = BootstrapStatic.model_validate(load("bootstrap_static.json")).elements[0]
    assert isinstance(player.selected_by_percent, float)
    assert isinstance(player.expected_goals, float)


def test_declared_drops_undeclared_fields():
    raw = load("bootstrap_static.json")["teams"][0]
    team = BootstrapStatic.model_validate(load("bootstrap_static.json")).teams[0]
    assert "pulse_id" in raw
    assert "pulse_id" not in team.declared()
    assert team.declared()["short_name"] == raw["short_name"]


def test_new_upstream_fields_are_tolerated():
    payload = load("bootstrap_static.json")
    payload["teams"][0]["brand_new_field"] = 1
    BootstrapStatic.model_validate(payload)


def test_missing_required_field_fails_loudly():
    payload = load("bootstrap_static.json")
    del payload["elements"][0]["now_cost"]
    with pytest.raises(ValidationError, match="now_cost"):
        BootstrapStatic.model_validate(payload)


def test_fixtures_parse_real_payload():
    fixtures = [Fixture.model_validate(f) for f in load("fixtures.json")]
    assert all(f.team_h != f.team_a for f in fixtures)


def test_element_summary_parses_real_payload():
    summary = ElementSummary.model_validate(load("element_summary.json"))
    assert summary.history
    assert all(isinstance(m.expected_goals, float) for m in summary.history)
