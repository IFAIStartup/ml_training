import os
import glob
import json
import time
import shutil
import datetime
import requests
import mlflow

import ml_training.dataset_processing as dp

from typing import Union

from mlflow.entities import ViewType
from ml_training.config import settings
from ml_training.pipeline.exception import NextcloudIsNotResponding
from ml_training.pipeline.utils import delete_file, delete_dir, create_dir


def get_dataset_nextcloud_service(id: str, path: str) -> Union[None, Exception]:
    os.makedirs(f"static/{id}", mode=0o777, exist_ok=True)

    url = f"{settings.NEXTCLOUD_URL}/{settings.NEXTCLOUD_API_PATH}{path}"
    response = requests.get(
        url, auth=(settings.NEXTCLOUD_LOGIN, settings.NEXTCLOUD_PASSWORD), stream=True
    )
    if response.status_code != 200:
        raise NextcloudIsNotResponding

    with open(f"static/{id}/{path}.zip", "wb", 8192) as file_handle:
        for chunk in response.iter_content(8192):
            file_handle.write(chunk)

    shutil.unpack_archive(f"static/{id}/{path}.zip", f"static/{id}/")
    os.remove(f"static/{id}/{path}.zip")


def upload_model_nextcloud_service(id: int, name_zip: str):
    file_handle = open(f"static/{id}/{name_zip}", "rb")
    url = f"{settings.NEXTCLOUD_URL}/remote.php/dav/files/airflow/{name_zip}"
    response = requests.put(
        url,
        auth=(settings.NEXTCLOUD_LOGIN, settings.NEXTCLOUD_PASSWORD),
        data=file_handle,
        stream=True,
    )
    if response.status_code != 201:
        raise NextcloudIsNotResponding

    delete_file(path=f"static/{id}/{name_zip}")


def delete_ml_model_nextcloud_service(path: str) -> Union[None, Exception]:
    url = f"{settings.NEXTCLOUD_URL}/index.php/apps/files/api/v1/file={path}"
    response = requests.delete(
        url, auth=(settings.NEXTCLOUD_LOGIN, settings.NEXTCLOUD_PASSWORD)
    )
    if response.status_code != 200:
        raise NextcloudIsNotResponding


def get_unique_experiment_name() -> str:
    exps = mlflow.search_experiments(ViewType.ALL)
    exp_name = f"experiment_{len(exps)}"
    return exp_name


def create_experiment(name: str, auth: tuple = None) -> requests.Response:
    data = {"name": name}
    url = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/experiments/create'
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    with requests.Session() as session:
        if auth:
            session.auth = auth
        res = session.post(url, headers=headers, data=json.dumps(data))

    return res


def create_unique_experiment_service(user: str, password: str) -> dict[str, str]:
    name_experiment = get_unique_experiment_name()
    response = create_experiment(name_experiment, (user, password))
    return response, name_experiment


def prepare_training_dataset(
        data_path: str, 
        classes: list[str], 
        crop_size: int,
        ml_model_type: str):

    preprocessed_data_path = os.path.join(data_path, "preprocessed_dataset")
    if os.path.exists(preprocessed_data_path):
        shutil.rmtree(preprocessed_data_path)

    overlapping = 0 # crop_size // 4

    if ml_model_type == 'yolov8':
        install_yolo_seg = True
        install_yolo_det = False
        install_masks = False
    
    elif ml_model_type == 'yolov8_det':
        install_yolo_seg = False
        install_yolo_det = True
        install_masks = False

    else:   #'deeplabv3'
        install_yolo_seg = False
        install_yolo_det = False
        install_masks = True
    

    create_dataset(
        src_dir=data_path,
        dst_dir=preprocessed_data_path,
        classes=classes,
        install_yolo_det=install_yolo_det,
        install_yolo_seg=install_yolo_seg,
        install_masks=install_masks,
        crop_size=crop_size,
        overlapping=overlapping,
    )


def create_dataset(
    src_dir: str,
    dst_dir: str,
    classes: list[str] = None,
    install_yolo_det: bool = False,
    install_yolo_seg: bool = True,
    install_masks: bool = True,
    crop_size: int = 1024,
    overlapping: int = 256,
):
    """Create dataset with the file structure that is suitable for training

    :param src_dir: directory with source datasets. Each one is a directory that contains folders `images` and `annotations`.
                    `annotations` contains `instances_default.json` file with source annotation in COCO format
    :param dst_dir: destination path to prepared dataset.
    :param classes: chosen class names for training, defaults to None (use all source classes)
    :param install_yolo_det: whether install yolo detection labels or not, defaults to False
    :param install_yolo_seg: whether install yolo segmentation labels or not, defaults to True
    :param install_masks: whether install segmentation masks or not, defaults to True
    :param crop_size: which crops size (in pixels) use for image cropping, 0 means do not crop images, defaults to 1024
    :param overlapping: crop overlapping (in pixels), defaults to 256
    """

    dataset = dp.Dataset()
    for i, cur_subdir in enumerate(os.listdir(src_dir)):
        src_images_dir = os.path.join(src_dir, cur_subdir, "images")
        src_annot_path = os.path.join(
            src_dir, cur_subdir, "annotations", "instances_default.json"
        )
        annot = dp.read_coco(src_annot_path)
        image_paths = glob.glob(os.path.join(src_images_dir, "*"))
        image_sources = dp.paths2image_sources(image_paths)

        cur_dataset = dp.Dataset(image_sources, annot)

        # TODO: add renaming
        if crop_size > 0:
            cur_dataset = dp.crop_dataset(
                cur_dataset, (crop_size, crop_size), overlapping
            )
        cur_dataset.annotation = change_annotation(cur_dataset.annotation, classes)
        cur_dataset.remove_empty_images(0.1)

        # TODO: check if we need special param for train/val proportions
        cur_dataset.split_by_proportions({"val": 0.2, "train": 0.8})

        dataset += cur_dataset
    dataset.install(
        dataset_path=dst_dir,
        dataset_name="dataset",
        install_images=True,
        install_yolo_det_labels=install_yolo_det,
        install_yolo_seg_labels=install_yolo_seg,
        install_coco_annotations=False,
        install_description=True,
        install_masks=install_masks,
        image_ext=".jpg",
    )

    # If there is too small amount of images for one of subsets
    # Then move one image
    val_dir = os.path.join(dst_dir, 'val')
    train_dir = os.path.join(dst_dir, 'train')
    val_images_dir = os.path.join(val_dir, 'images')
    train_images_dir = os.path.join(train_dir, 'images')
    if len(os.listdir(val_images_dir)) == 0 and len(os.listdir(train_images_dir)) != 0:
        shutil.rmtree(val_dir)
        shutil.copytree(train_dir, val_dir)
    elif len(os.listdir(val_images_dir)) != 0 and len(os.listdir(train_images_dir)) == 0:
        shutil.rmtree(train_dir)
        shutil.copytree(val_dir, train_dir)

    if len(os.listdir(train_images_dir)) == 1:
        fn = os.listdir(train_images_dir)[0]
        name, ext = os.path.splitext(fn)
        shutil.copy(
            os.path.join(train_dir, 'images', fn), 
            os.path.join(train_dir, 'images', f"{name}_copy{ext}")
        )
        shutil.copy(
            os.path.join(train_dir, 'labels', f"{name}.txt"), 
            os.path.join(train_dir, 'labels', f"{name}_copy.txt")
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


def create_specific_run(
    experiment_id: str, model_name: str, user: str, password: str
) -> str | None:
    run_name = get_timestamped_run_name(model_name=model_name)
    res = create_run(
        experiment_id=experiment_id, run_name=run_name, auth=(user, password)
    )
    if res.status_code != 200:
        return None

    run_id = json.loads(res.text)["run"]["info"]["run_uuid"]
    return run_id


def get_timestamped_run_name(model_name: str) -> str:
    strftime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_run_{strftime}"
    return run_name


def create_run(
    experiment_id: str, run_name: str, auth: tuple = None
) -> requests.Response:
    start_time = int(time.time() * 1000)
    data = {
        "experiment_id": experiment_id,
        "run_name": run_name,
        "start_time": start_time,
    }
    url = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/runs/create'
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    with requests.Session() as session:
        if auth:
            session.auth = auth
        res = session.post(url, headers=headers, data=json.dumps(data))

    return res


def check_experiment_exists_service(path: str) -> bool:
    if os.path.exists(path):
        return True
    return False


def format_dir_service(id: int, experiment_path: str, experiment_name: str) -> None:
    path = f"static/{id}/{experiment_name}"
    shutil.copytree(experiment_path, path, dirs_exist_ok=True)
    models = glob.glob(f"{path}/yolov8*")
    if not models:
        return load_deeplab_model_service(path=path)
    models.sort()
    model = models[-1]
    for i in glob.glob(f"{model}/*"):
        if "weights" not in i and "config.pbtxt" not in i:
            delete_file(path=i)
    os.rename(f"{model}/weights", f"{model}/1")

    for i in glob.glob(f"{model}/1/*"):
        if "best.onnx" not in i:
            delete_file(path=i)
    os.rename(f"{model}/1/best.onnx", f"{model}/1/model.onnx")

    for i in glob.glob(f"{path}/*"):
        if i != model:
            delete_dir(path=i)


def load_deeplab_model_service(path: str) -> str:
    models = glob.glob(f"{path}/deeplabv3*")
    models.sort()
    model = models[-1]
    
    for i in glob.glob(f"{model}/*"):
        if "model.onnx" not in i and "config.pbtxt" not in i:
            try:
                delete_file(path=i)
            except IsADirectoryError:
                delete_dir(path=i)
    create_dir(path=f"{model}/1")
    shutil.move(f"{model}/model.onnx", f"{model}/1/model.onnx")

    for i in glob.glob(f"{path}/*"):
        if i != model:
            delete_dir(path=i)
