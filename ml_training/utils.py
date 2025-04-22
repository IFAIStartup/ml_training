import os
import json
import time
import requests
import logging
import datetime
import glob
import shutil
from typing import List, Tuple, Dict, Any
from enum import Enum
import onnx
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.entities import ViewType
import ml_training.dataset_processing as dp

logger = logging.getLogger(__name__)


class DataType(Enum):
    TYPE_FP32 = 0
    TYPE_FP16 = 1


DATATYPE_TABLE = {
    DataType.TYPE_FP16: "TYPE_FP16",
    DataType.TYPE_FP32: "TYPE_FP32",
}


def get_project_and_run_name(model_name: str) -> Tuple[str, str]:
    """Create project name and run name by provided model name

    :param model_name: name of chosen architecture of ml model (yolov8, deeplabv3, ...)
    :return: tuple of 
             - project name 
             - run name 
    """
    project_name = model_name
    strftime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{project_name}_run_{strftime}" 
    return project_name, run_name


def get_unique_experiment_name() -> str:
    exps = mlflow.search_experiments(ViewType.ALL)
    exp_name = f'experiment_{len(exps)}'
    return exp_name

def get_timestamped_run_name(model_name: str) -> str:
    strftime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_name}_run_{strftime}" 
    return run_name


def start_training_run(model_name: str, exp_name: str) -> str:
    """Start training run for MLFLow Server with provided model name

    :param model_name: name of chosen architecture of ml model (yolov8, deeplabv3, ...)
    :return: run id
    """
    mlflow.end_run()
    run_name = get_timestamped_run_name(model_name)
    mlflow.set_experiment(exp_name)
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    return run_id


def create_unique_experiment(user: str, password: str):
    exp_name = get_unique_experiment_name()
    res = create_experiment(exp_name, (user, password))
    return res, exp_name


def create_specific_run(experiment_id: str, model_name: str, user: str, password: str) -> str | None:
    run_name = get_timestamped_run_name(model_name)
    res = create_run(experiment_id, run_name, (user, password))
    if res.status_code != 200:
        return None
    
    run_id = json.loads(res.text)['run']['info']['run_uuid']
    return run_id


def create_experiment(name: str, auth: tuple = None) -> requests.Response:
    data = {"name": name}
    url = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/experiments/create'
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    with requests.Session() as session:
        if auth:
            session.auth = auth
        res = session.post(url, headers=headers, data=json.dumps(data))

    return res


def create_run(experiment_id: str, run_name: str, auth: tuple = None) -> requests.Response:
    start_time = int(time.time() * 1000)
    data = {"experiment_id": experiment_id, "run_name": run_name, "start_time": start_time}
    url = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/runs/create'
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    with requests.Session() as session:
        if auth:
            session.auth = auth
        res = session.post(url, headers=headers, data=json.dumps(data))

    return res


def log_and_register_mlflow_model(
        onnx_path: str, 
        dummy_ins: Dict[str, Any], 
        dummy_outs: Dict[str, Any],
        registered_model_name: str,
        ):
    onnx_model = onnx.load_model(onnx_path)
    signature = infer_signature(dummy_ins, dummy_outs)
    
    model_info = mlflow.onnx.log_model(onnx_model, "model_onnx", signature=signature)
    
    if registered_model_name:
        mv = mlflow.register_model(model_info.model_uri, registered_model_name)
        client = MlflowClient()
        client.set_registered_model_alias(registered_model_name, 'last', mv.version)

        # TODO: Add "best" alias



def dict_to_pbtxt(src: dict) -> str:
    return _convert_to_pbtxt(src)


def _convert_to_pbtxt(src: Any, level: int = 0) -> str:
    if type(src) == dict:
        pairs = []
        for key in src:
            pair = level * '    ' + f"{key}: {_convert_to_pbtxt(src[key], level + 1)}"
            pairs.append(pair)
        string = '\n'.join(pairs)
        if level > 0:
            string = "{\n" + string + "\n" + (level - 1) * '    ' + "}"
        return string

    if type(src) == list:
        elems = []
        for elem in src:
            elems.append(_convert_to_pbtxt(elem, level + 1))
        string = ', '.join(elems)
        string = "[ " + string  + " ]"
        return string

    if type(src) == DataType:
        return DATATYPE_TABLE[src]

    if type(src) == str:
        return "\"" + str(src) + "\""

    return str(src)


def prepare_training_dataset(data_path: str, classes: List[str], crop_size: int):
    preprocessed_data_path = os.path.join(data_path, 'preprocessed_dataset')
    if os.path.exists(preprocessed_data_path):
        shutil.rmtree(preprocessed_data_path)

    install_yolo_seg = True 
    install_masks = True
    overlapping = crop_size // 4
    
    create_dataset(
        data_path, 
        preprocessed_data_path, 
        classes, 
        install_yolo_seg, 
        install_masks, 
        crop_size, 
        overlapping)



def create_dataset(
        src_dir: str, 
        dst_dir: str, 
        classes: List[str] = None,
        install_yolo_seg: bool = True, 
        install_masks: bool = True,
        crop_size: int = 1024,
        overlapping: int = 256,):
    """Create dataset with the file structure that is suitable for training

    :param src_dir: directory with source datasets. Each one is a directory that contains folders `images` and `annotations`.
                    `annotations` contains `instances_default.json` file with source annotation in COCO format
    :param dst_dir: destination path to prepared dataset.
    :param classes: chosen class names for training, defaults to None (use all source classes)
    :param install_yolo_seg: whether install yolo segmentation labels or not, defaults to True
    :param install_masks: whether install segmentation masks or not, defaults to True
    :param crop_size: which crops size (in pixels) use for image cropping, 0 means do not crop images, defaults to 1024
    :param overlapping: crop overlapping (in pixels), defaults to 256
    """

    dataset = dp.Dataset()
    for i, cur_subdir in enumerate(os.listdir(src_dir)):
        src_images_dir = os.path.join(src_dir, cur_subdir, 'images')
        src_annot_path = os.path.join(src_dir, cur_subdir, 'annotations', 'instances_default.json')
        
        annot = dp.read_coco(src_annot_path)
        image_paths = glob.glob(os.path.join(src_images_dir, '*'))
        image_sources = dp.paths2image_sources(image_paths)
        
        cur_dataset = dp.Dataset(image_sources, annot)
        
        # TODO: add renaming
        if crop_size > 0:
            cur_dataset = dp.crop_dataset(cur_dataset, (crop_size, crop_size), overlapping)
        cur_dataset.annotation = change_annotation(cur_dataset.annotation, classes)
        cur_dataset.remove_empty_images(0.1)

        # TODO: check if we need special param for train/val proportions
        cur_dataset.split_by_proportions({'val': 0.2, 'train': 0.8})
        
        dataset += cur_dataset

    
    dataset.install(
        dataset_path=dst_dir,
        dataset_name='dataset',
        install_images=True,
        install_yolo_seg_labels=install_yolo_seg,
        install_coco_annotations=False,
        install_description=True,
        install_masks = install_masks,
        image_ext='.jpg',
    )
    

def change_annotation(annot: dp.Annotation, new_classes: list):
    classes = annot.categories
    
    conformity = {}
    for i in range(len(classes)):
        if classes[i] in new_classes:
            conformity[i] = new_classes.index(classes[i])
    
    images = annot.images
    for name in images:
        new_bboxes = []
        
        for bbox in images[name].annotations:
            if bbox.category_id not in conformity:
                continue
            
            bbox.category_id = conformity[bbox.category_id]
            new_bboxes.append(bbox)
        images[name].annotations = new_bboxes
        
    annot.categories = new_classes
    return annot
    
