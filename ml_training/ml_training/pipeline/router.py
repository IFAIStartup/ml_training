import json
from typing import Dict
import time
from fastapi import APIRouter, HTTPException, status
from celery.result import AsyncResult

from ml_training.pipeline.service import (
    create_unique_experiment_service,
    check_experiment_exists_service,
)
from ml_training.pipeline.worker import (
    get_dataset_nextcloud,
    train_ml_models,
    upload_model_nextcloud,
)
from ml_training.pipeline.schemas import GetTaskResultSchemas, StartTrainSchemas


router = APIRouter(
    prefix="/pipline",
    tags=["pipline"],
)


@router.get("/task-status/{task_id}")
async def train_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    return GetTaskResultSchemas(
        task_id=task_result.task_id,
        task_status=task_result.status,
        task_result=task_result.result,
    )


@router.post("/upload-dataset")
async def dwonload_dataset(id: int, path: str) -> Dict[str, str]:
    result = get_dataset_nextcloud.apply_async(args=(id, path))
    return {"task_id": result.id}


@router.post("/train")
async def train(params: StartTrainSchemas) -> Dict[str, str]:
    response, name_experiment = create_unique_experiment_service(
        params.creds.user, params.creds.password
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    experiment_id = json.loads(response.text)["experiment_id"]
    runs_info = [json.loads(run_info.model_dump_json()) for run_info in params.rus_info]
    result = train_ml_models.delay(
        runs_info,
        experiment_id,
        name_experiment,
        params.creds.user,
        params.creds.password,
    )

    experiment_value = "empty"
    task_id = result.id
    result = AsyncResult(task_id)

    while not result.ready():
        result = AsyncResult(task_id)
        if not result.ready():
            state = result.state
            if state == "PROGRESS":
                info = result.info
                experiment_value = info.get("experiment_value")
                break
        time.sleep(1)

    return {
        "task_id": result.id,
        "experiment_id": experiment_id,
        "experiment_name": name_experiment,
        "experiment_value": experiment_value,
    }


@router.post("/save-ml-model")
async def save_ml_model(id: int, experiment_name: str) -> Dict[str, str]:
    path = f"/ml_training/experiments/{experiment_name}"
    if not check_experiment_exists_service(path=path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Experiment not found"
        )
    result = upload_model_nextcloud.delay(id, path, experiment_name)
    return {
        "task_id": result.id,
    }
