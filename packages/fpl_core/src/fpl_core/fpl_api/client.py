"""Client for the public (undocumented) FPL API.

Concurrency is capped so a full refresh of every player stays polite.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fpl_core.http import HttpError, RetryingHttpClient

DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"

FplApiError = HttpError


class FplApiClient(RetryingHttpClient):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        max_workers: int = 4,
        **kwargs: Any,  # noqa: ANN401 - forwarded to RetryingHttpClient
    ) -> None:
        super().__init__(base_url, **kwargs)
        self._max_workers = max_workers

    def get_json(self, path: str) -> Any:  # noqa: ANN401 - raw JSON is validated by callers
        return self.get(path).json()

    def bootstrap_static(self) -> dict[str, Any]:
        """Teams, gameweeks, positions and every player's current state."""
        result: dict[str, Any] = self.get_json("/bootstrap-static/")
        return result

    def fixtures(self) -> list[dict[str, Any]]:
        """All fixtures of the current season, including per-fixture event stats."""
        result: list[dict[str, Any]] = self.get_json("/fixtures/")
        return result

    def element_summary(self, element_id: int) -> dict[str, Any]:
        """One player's match history this season, past seasons and upcoming fixtures."""
        result: dict[str, Any] = self.get_json(f"/element-summary/{element_id}/")
        return result

    def element_summaries(self, element_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = list(element_ids)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return dict(zip(ids, pool.map(self.element_summary, ids), strict=True))

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        """A manager's season so far: per-gameweek bank, value, transfers, chips."""
        result: dict[str, Any] = self.get_json(f"/entry/{entry_id}/history/")
        return result

    def entry_transfers(self, entry_id: int) -> list[dict[str, Any]]:
        """Every transfer this season, with the price paid (`element_in_cost`)."""
        result: list[dict[str, Any]] = self.get_json(f"/entry/{entry_id}/transfers/")
        return result

    def entry_picks(self, entry_id: int, gameweek: int) -> dict[str, Any]:
        """The 15 players a manager held for `gameweek`, plus that week's bank."""
        result: dict[str, Any] = self.get_json(f"/entry/{entry_id}/event/{gameweek}/picks/")
        return result
