from celery import Celery
from ml_training.config import settings


app_worker = Celery(
    __name__,
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
)
