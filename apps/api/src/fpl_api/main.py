from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fpl_api.routes import build_router
from fpl_api.settings import Settings, get_settings
from fpl_api.store import Store


class Health(BaseModel):
    status: str
    environment: str
    version: str


def create_app(store: Store | None = None) -> FastAPI:
    settings = get_settings()
    data = store or Store(settings.warehouse, settings.lake)
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

    app.include_router(build_router(lambda: data))
    return app


app = create_app()
