from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from ml_training.routers import routers

__version__ = "0.1.0"


def create_app() -> Any:
    app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    app.mount("/static", StaticFiles(directory="static"), name="static")

    routers(app=app)
    return app
