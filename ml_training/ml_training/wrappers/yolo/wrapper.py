import os
from typing import Tuple
import onnx
import mlflow
from mlflow.models import infer_signature
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import yaml
import datetime
import ultralytics
from ml_training.wrappers.yolo.callbacks import YOLOCallback
from ml_training.wrappers.base_wrapper import BaseWrapper
from ml_training.utils import get_project_and_run_name

# TODO: log model pt

# Disable default mlflow and wandb logging
ultralytics.settings.update({"wandb": False})
ultralytics.settings.update({"mlflow": False})


class YoloTrainWrapper(BaseWrapper):

    def __init__(
        self,
        config: dict,
        registered_model_name: str = None,
        run_id: str = None,
        path_data_yaml: str = ""
    ):
        super().__init__(config, registered_model_name, run_id)

        self.data = config["data"]
        self.imgsz = config["imgsz"]
        self.config["project"] = os.path.join("experiments", self.project_name)
        self.config["name"] = self.run_name
        self.path_data_yaml = path_data_yaml
        self._fix_yolo_dataset_cfg(self.path_data_yaml)

        self.model = YOLO(config["model"])

        # Add the custom callbacks to the model
        self.custom_callback = YOLOCallback(config, registered_model_name, run_id)
        self.model.add_callback(
            "on_fit_epoch_end", self.custom_callback.on_fit_epoch_end
        )
        self.model.add_callback(
            "on_pretrain_routine_end", self.custom_callback.on_pretrain_routine_end
        )
        self.model.add_callback("on_train_end", self.custom_callback.on_train_end)
        self.model.add_callback(
            "on_train_epoch_end", self.custom_callback.on_train_epoch_end
        )

    def train(self, with_pbtxt: bool = False):
        self.model.train(**self.config)
        if with_pbtxt:
            self.create_pbtxt()

    def validate(self):
        self.model.val(**self.config)

    def create_pbtxt(self):
        # Get path to onnx model
        onnx_path = str(self.model.trainer.best).replace('.pt', '.onnx')
        
        # Get random input and output arrays to find their sizes for config.pbtxt
        dummy_in, outs = self.custom_callback._get_dummy_ins_outs(onnx_path)
        
        # Create config.pbtxt
        self._save_pbtxt(outs)

    def _save_pbtxt(self, outs):
        path = os.path.join(self.config["project"], self.config["name"], "config.pbtxt")

        output_configs = []
        for i, out in enumerate(outs):
            out_cfg = (
                "{\n"
                f'    name: "output{i}"\n'
                "    data_type: TYPE_FP32\n"
                "    dims: "
                f"[ {out.shape[0]}, {out.shape[1]}, {out.shape[2]} ]"
                "\n"
                "}"
            )
            output_configs.append(out_cfg)
        output_config_txt = ',\n'.join(output_configs)

        pbtxt = (
            f'name: "{self.run_name}"\n'
            'platform: "onnxruntime_onnx"\n'
            "max_batch_size : 0\n"
            "input [\n"
            "{\n"
            '    name: "images"\n'
            "    data_type: TYPE_FP32\n"
            "    dims: "
            f"[ 3, {self.imgsz}, {self.imgsz} ]"
            "\n"
            "    reshape { shape: "
            f"[ 1, 3, {self.imgsz}, {self.imgsz} ]"
            " }\n"
            "}\n"
            "]\n"
            "output [\n"
            f"{output_config_txt}\n"
            "]\n"
        )
        
        with open(path, "w") as f:
            f.write(pbtxt)

    def _fix_yolo_dataset_cfg(self, dataset_cfg_path: str):
        """Paste correct paths to .yaml config file (usually its `data.yaml`)
        that places in root dataset directory

        :param dataset_cfg_path: path to .yaml file
        """
        dataset_dir = os.path.dirname(dataset_cfg_path)
        train_images_dir = os.path.join(dataset_dir, "train", "images")
        val_images_dir = os.path.join(dataset_dir, "val", "images")

        with open(dataset_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["train"] = train_images_dir
        cfg["val"] = val_images_dir

        with open(dataset_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

    def _get_project_and_run_name(self) -> Tuple[str]:
        return get_project_and_run_name("yolov8")
