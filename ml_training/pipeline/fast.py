import os
import json
import logging

from celery.result import AsyncResult
from mlflow.server import get_app_client
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from ml_training.utils import create_unique_experiment
from ml_training.pipeline.tasks import train_ml_models, get_dataset_nextcloud
from ml_training.deployment.deployment_client import TritonDeploymentClient
from mlflow.exceptions import RestException

logger = logging.getLogger(__name__)

app = FastAPI()

# Set the MLflow tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

try:
    # Create test users with credentials for authentication
    auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
    auth_client.create_user(username="user1", password="pw1")
    auth_client.create_user(username="user2", password="pw2")
except RestException as e:
    print(e)


class Credentials(BaseModel):
    user: str
    password: str


class RunInfo(BaseModel):
    data_path: str = (
        "/app/datasets/aerial_example_datasets"  # Path to dataset with specific file structure
    )
    classes: List[str] = ["building"]  # Specific classes on which model will be trained
    crop_size: int = (
        1024  # Size of image crop, that will be used while image splitting, 0 means do not split
    )
    ml_model_type: str = "yolov8"  # Type of training ML model ("yolov8" or "deeplabv3")
    img_size: int = (
        640  # Size of the model input images (input image has square shapes (img_size, img_size))
    )
    epochs: int = 1  # Number of training epochs
    registered_model_name: str | None = (
        None  # Name which will be used for trained model in Model Registry, None means the model won't be registered
    )
    config: dict | None = (
        None  # Extra config for training. If provided, replaces every provided key in default configuration
    )
"""
{
  "creds": {
    "user": "admin",
    "password": "password"
  },
  "rus_info": [
    {
      "data_path": "1/test_dataset",
      "classes": [
        "building"
      ],
      "crop_size": 1024,
      "ml_model_type": "yolov8",
      "img_size": 640,
      "epochs": 1,
      "registered_model_name": "",
      "config": {}
    }
  ]
}
"""

class ExperimentInfo(BaseModel):
    creds: Credentials
    runs_info: List[RunInfo] = [
        RunInfo(ml_model_type="yolov8"),
        RunInfo(ml_model_type="deeplabv3"),
    ]


@app.get("/train-task-status/{task_id}")
async def train_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    status = task_result.status
    return status


@app.post("/upload-dataset")
async def dwonload_dataset(id: int, path: str) -> Dict[str, str]:
    result = get_dataset_nextcloud.delay(id, path)
    return {"celery-task-id": result.id}


@app.post("/train/")
async def train(item: ExperimentInfo):
    logging.info("Start train")
    user = item.creds.user
    password = item.creds.password
    runs_info = item.runs_info

    # Try to create unique experiment with authentification
    # If something went wrong (creds are invalid, for example), throw exception
    res, exp_name = create_unique_experiment(user, password)
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)

    experiment_id = json.loads(res.text)["experiment_id"]
    runs_info = [json.loads(run_info.model_dump_json()) for run_info in runs_info]

    # Start Celery task for training
    result = train_ml_models.delay(runs_info, experiment_id, exp_name, user, password)
    # return {"mlflow-experiment-name": exp_name, "celery-task-id": result.id}
    return {"mlflow-experiment-name": "exp_name", "celery-task-id": result.id}


@app.get("/deployment/list")
async def deployment_list():
    deployment_client = TritonDeploymentClient(
        "/models", "http://triton_server:" + os.getenv("TRITON_PORT")
    )
    deployments = deployment_client.list_deployments()
    return deployments


@app.post("/deployment/create/")
async def deployment_create(name: str, model_uri: str):
    deployment_client = TritonDeploymentClient(
        "/models", "http://triton_server:" + os.getenv("TRITON_PORT")
    )
    deployment_client.create_deployment(name, model_uri)
    return deployment_client.list_deployments()


@app.post("/deployment/delete/")
async def deployment_create(name: str):
    deployment_client = TritonDeploymentClient(
        "/models", "http://triton_server:" + os.getenv("TRITON_PORT")
    )
    deployment_client.delete_deployment(name)
    return deployment_client.list_deployments()
