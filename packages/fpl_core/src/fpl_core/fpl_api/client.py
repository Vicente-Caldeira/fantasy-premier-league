"""HTTP client for the public (undocumented) FPL API.

Retries transient failures with exponential backoff, honours `Retry-After`, and caps
concurrency so a full refresh stays polite to a service we don't own.
"""

import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self

import httpx

DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_USER_AGENT = "fpl-xpts/0.1 (+https://github.com/Vicente-Caldeira/fantasy-premier-league)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class FplApiError(RuntimeError):
    """The API returned a non-retryable error or retries were exhausted."""


class FplApiClient:
    def __init__(  # noqa: PLR0913 - explicit knobs beat a config object here
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        max_workers: int = 4,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            transport=transport,
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_workers = max_workers
        self._sleep = sleep

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str) -> Any:  # noqa: ANN401 - raw JSON is validated by callers
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(path)
            except httpx.TransportError as err:
                if attempt == self._max_retries:
                    raise FplApiError(f"GET {path}: {err!r} after {attempt} retries") from err
                self._sleep(self._delay(attempt, None))
                continue

            if response.status_code == httpx.codes.OK:
                return response.json()
            if response.status_code not in RETRYABLE_STATUS or attempt == self._max_retries:
                raise FplApiError(
                    f"GET {path}: HTTP {response.status_code} after {attempt} retries"
                )
            self._sleep(self._delay(attempt, response.headers.get("Retry-After")))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None and retry_after.isdigit():
            return float(retry_after)
        return float(self._backoff_seconds * 2**attempt)

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
