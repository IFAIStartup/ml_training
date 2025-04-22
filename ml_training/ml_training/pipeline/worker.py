import time

from ml_training.worker import app_worker
from ml_training.pipeline.service import (
    get_dataset_nextcloud_service,
    upload_model_nextcloud_service,
    prepare_training_dataset,
    create_specific_run,
    format_dir_service,
)
from ml_training.pipeline.utils import zip_directory, delete_dir, delete_file
from ml_training.pipeline.train import train_specified_model
from ml_training.pipeline.exception import NextcloudIsNotResponding


@app_worker.task(name="get_dataset_nextcloud")
def get_dataset_nextcloud(id: int, path: str) -> dict[str, str]:
    try:
        get_dataset_nextcloud_service(id=id, path=path)
    except NextcloudIsNotResponding:
        return {"status": "NextcloudIsNotResponding"}
    return {"status": "ok"}


@app_worker.task(name="train_ml_models", bind=True, ignore_result=False)
def train_ml_models(
    self, runs_info: list[dict], experiment_id: str, exp_name: str, user: str, password: str
) -> tuple[list[str], list[str]]:

    result = []
    for index, run_info in enumerate(runs_info):

        # Check if ML model type is valid
        # if run_info["ml_model_type"] not in ["yolov8", "deeplabv3"]:
        #    continue

        path = f"static/{run_info['data_path']}"

        # Create special dataset that is suitable for training
        prepare_training_dataset(
            data_path=path, 
            classes=run_info["classes"], 
            crop_size=run_info["crop_size"], 
            ml_model_type=run_info["ml_model_type"],
        )

        # Create run in MLflow
        # If creating wasn't successful, just skip this run

        run_id = create_specific_run(
            experiment_id=experiment_id,
            model_name=run_info["ml_model_type"],
            user=user,
            password=password,
        )
        if run_id is None:
            continue

        if index == 0:
            self.update_state(state="PROGRESS", meta={"experiment_value": run_id})

        # Start training loop with logging to previously created run
        train_specified_model(run_info=run_info, run_id=run_id, static_path="static")
        result.append(run_id)

    return result[0]


@app_worker.task(name="upload_model_nextcloud")
def upload_model_nextcloud(id: int, path: str, experiment_name) -> None:
    format_dir_service(id=id, experiment_path=path, experiment_name=experiment_name)
    zip_directory(
        folder_path=f"static/{id}/{experiment_name}",
        zip_file=f"static/{id}/{experiment_name}",
    )
    delete_dir(path=f"static/{id}/{experiment_name}")
    upload_model_nextcloud_service(id=id, name_zip=f"{experiment_name}.zip")
    delete_file(f"static/{id}/{experiment_name}.zip")
