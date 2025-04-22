from typing import Any

from pydantic import BaseModel


class GetTaskResultSchemas(BaseModel):
    task_id: str
    task_status: str
    task_result: Any


class CredentialsSchemas(BaseModel):
    user: str
    password: str


class RunInfo(BaseModel):
    data_path: str
    classes: list[str]
    crop_size: int
    ml_model_type: str
    img_size: int
    epochs: int
    registered_model_name: str | None = None
    config: dict | None = None


class StartTrainSchemas(BaseModel):
    creds: CredentialsSchemas
    rus_info: list[RunInfo]
