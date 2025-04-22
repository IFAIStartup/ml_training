import os
import requests
import shutil
from celery import Celery
from celery.result import AsyncResult
from typing import List
from dotenv import dotenv_values
from ml_training.pipeline.train import train_specified_model
from ml_training.utils import prepare_training_dataset, create_specific_run


config = {
    **dotenv_values(".env"),  # load general environment variables
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

app = Celery(
    "ml_training.pipeline.tasks",
    broker=config["CELERY_BROKER"],
    backend=config["CELERY_BACKEND"],
    include=["ml_training.pipeline.tasks"],
)


@app.task
def get_dataset_nextcloud(id: int, path: str):
    os.makedirs(f"/app/datasets/{id}", mode=0o777, exist_ok=True)
    nextcloud_url = os.getenv("NEXTCLOUD_URL")
    nextcloud_path = "index.php/apps/files/ajax/download.php?dir="
    url = f"{nextcloud_url}/{nextcloud_path}{path}"
    response = requests.get(url, auth=("Admin123", "Work@123"), stream=True)
    file_handle = open(f"/app/datasets/{id}/{path}.zip", "wb", 8192)
    for chunk in response.iter_content(8192):
        file_handle.write(chunk)
    file_handle.close()

    shutil.unpack_archive(f"/app/datasets/{id}/{path}.zip", f"/app/datasets/{id}/")

    os.remove(f"/app/datasets/{id}/{path}.zip")


@app.task
def train_ml_model(item: dict, run_id: str = None):
    prepare_training_dataset(item["data_path"], item["classes"], item["crop_size"])
    train_specified_model(item, run_id)


@app.task
def train_ml_models(
    runs_info: List[dict], experiment_id: str, exp_name: str, user: str, password: str
):
    for run_info in runs_info:

        # Check if ML model type is valid
        if run_info["ml_model_type"] not in ["yolov8", "deeplabv3"]:
            continue

        data_path = f"/app/datasets/{run_info['data_path']}"

        # Create special dataset that is suitable for training
        prepare_training_dataset(data_path, run_info["classes"], run_info["crop_size"])

        # Create run in MLflow
        # If creating wasn't successful, just skip this run
        run_id = create_specific_run(
            experiment_id, run_info["ml_model_type"], user, password
        )
        if run_id is None:
            continue

        # Start training loop with logging to previously created run
        train_specified_model(run_info, run_id)


### The code below is just testing celery tasks


@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)
