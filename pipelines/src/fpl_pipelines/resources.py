import dagster as dg

from fpl_core.fpl_api import DEFAULT_BASE_URL, FplApiClient


class FplApiResource(dg.ConfigurableResource):  # type: ignore[type-arg]
    base_url: str = DEFAULT_BASE_URL
    max_workers: int = 4

    def client(self) -> FplApiClient:
        return FplApiClient(self.base_url, max_workers=self.max_workers)
