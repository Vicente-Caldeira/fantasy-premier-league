from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fpl_api.settings import Settings, get_settings


class Health(BaseModel):
    status: str
    environment: str
    version: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="FPL xPts API", version=settings.version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["ops"])
    def healthz(settings: Annotated[Settings, Depends(get_settings)]) -> Health:
        return Health(status="ok", environment=settings.environment, version=settings.version)

    return app


app = create_app()
