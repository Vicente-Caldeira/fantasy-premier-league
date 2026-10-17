"""Polite HTTP client shared by every data source.

Retries transient failures with exponential backoff and honours `Retry-After`, so a
refresh never hammers a service we don't own.
"""

import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Self

import httpx

DEFAULT_USER_AGENT = "fpl-xpts/0.1 (+https://github.com/Vicente-Caldeira/fantasy-premier-league)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpError(RuntimeError):
    """A non-retryable error was returned, or retries were exhausted."""


class RetryingHttpClient:
    def __init__(  # noqa: PLR0913 - explicit knobs beat a config object here
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": user_agent, **(headers or {})},
            transport=transport,
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
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

    def get(self, path: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(path)
            except httpx.TransportError as err:
                if attempt == self._max_retries:
                    raise HttpError(f"GET {path}: {err!r} after {attempt} retries") from err
                self._sleep(self._delay(attempt, None))
                continue

            if response.status_code == httpx.codes.OK:
                return response
            if response.status_code not in RETRYABLE_STATUS or attempt == self._max_retries:
                raise HttpError(f"GET {path}: HTTP {response.status_code} after {attempt} retries")
            self._sleep(self._delay(attempt, response.headers.get("Retry-After")))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None and retry_after.isdigit():
            return float(retry_after)
        return float(self._backoff_seconds * 2**attempt)
