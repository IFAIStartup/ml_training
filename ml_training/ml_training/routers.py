from fastapi import FastAPI

from ml_training.pipeline.router import router as pipline_router


def routers(app: FastAPI) -> None:
    app.include_router(router=pipline_router, prefix="/api")
