import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings(BaseSettings):
    """Global config."""

    # RUN
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8080))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "debug")
    RELOAD: bool = bool(os.getenv("RELOAD", True))
    DEBUG: bool = bool(os.getenv("DEBUG", True))

    # CELERY
    CELERY_BROKER: str = os.getenv("CELERY_BROKER", "")
    CELERY_BACKEND: str = os.getenv("CELERY_BACKEND", "")

    # NEXTCLOUD_URL
    NEXTCLOUD_URL: str = os.getenv("NEXTCLOUD_URL", "")
    NEXTCLOUD_LOGIN: str = os.getenv("NEXTCLOUD_LOGIN", "")
    NEXTCLOUD_PASSWORD: str = os.getenv("NEXTCLOUD_PASSWORD", "")
    NEXTCLOUD_API_PATH: str = os.getenv("NEXTCLOUD_API_PATH", "")


settings = Settings()

