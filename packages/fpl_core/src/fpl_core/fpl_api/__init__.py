"""Client and data contracts for the public FPL API."""

from fpl_core.fpl_api.client import DEFAULT_BASE_URL, FplApiClient, FplApiError
from fpl_core.fpl_api.schemas import (
    BootstrapStatic,
    ElementSummary,
    Fixture,
    Gameweek,
    Player,
    PlayerMatch,
    Position,
    Team,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "BootstrapStatic",
    "ElementSummary",
    "Fixture",
    "FplApiClient",
    "FplApiError",
    "Gameweek",
    "Player",
    "PlayerMatch",
    "Position",
    "Team",
]
