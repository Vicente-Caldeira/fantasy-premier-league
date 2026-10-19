from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPL_API_", env_file=".env", extra="ignore")

    environment: str = "local"
    version: str = "dev"
    cors_origins: list[str] = ["http://localhost:3000"]
    # Read-only data sources, relative to the working directory (the repo root).
    warehouse: Path = Path("data/warehouse.duckdb")
    lake: str = "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
