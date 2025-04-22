import logging

import uvicorn

from ml_training import create_app
from ml_training.config import settings

logger = logging.getLogger(__name__)

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        reload=settings.RELOAD,
    )
