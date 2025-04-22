import os
import datetime
from typing import Tuple
import cv2
import mlflow
from mlflow import MlflowClient
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from ml_training.wrappers.deeplab.callbacks import PostTrainCallback
from ml_training.wrappers.deeplab.dataset import SegmentationDataset
from ml_training.wrappers.deeplab.tools import find_images
from ml_training.wrappers.deeplab.image_processing import (
    preproc_image,
    preproc_mask,
    get_augmentations,
)
from ml_training.wrappers.deeplab.model import DeepLabV3PlusModule
from ml_training.wrappers.deeplab.utils import COLOR_PALLET
from ml_training.wrappers.base_wrapper import BaseWrapper
from ml_training.utils import get_project_and_run_name


class DeepLabTrainWrapper(BaseWrapper):
    def __init__(
        self,
        config: dict,
        registered_model_name: str = None,
        run_id: str = None,
    ):
        super().__init__(config, registered_model_name, run_id)

        # Setting up dataset directories
        dataset_dir = os.path.join(
            os.path.dirname(__file__), self.config["DATA"]["dataset_dir"]
        )
        self.train_dataset_dir = os.path.join(dataset_dir, "train")
        self.val_dataset_dir = os.path.join(dataset_dir, "val")

        # Setup procedures for training
        self._setup_dataloader()
        self._setup_model()
        self._setup_callbacks()
        self._setup_logger()
        self._setup_trainer()

    def train(self, with_pbtxt: bool = False):

        # Start training and validation process
        self._train()
        if with_pbtxt:
            self.create_pbtxt()
        self.validate()

        if self.config['TRAIN']['valid']:
            self.__draw_results()
    
    
    def create_pbtxt(self):
        self._save_pbtxt()

    def _save_pbtxt(self):
        path = os.path.join('experiments', self.project_name, self.run_name, "config.pbtxt")
        img_size = self.config["DATA"]["img_size"]
        pbtxt = (
            f'name: "{self.run_name}"\n'
            'platform: "onnxruntime_onnx"\n'
            "max_batch_size : 0\n"
            "input [\n"
            "{\n"
            '    name: "input"\n'
            "    data_type: TYPE_FP32\n"
            "    dims: "
            f"[ 1, 3, {img_size}, {img_size} ]"
            "}\n"
            "]\n"
        )

        with open(path, "w") as f:
            f.write(pbtxt)

    def validate(self):
        # Validation process
        self.trainer.validate(self.model, self.val_loader)

    def _train(self):
        # Training process
        mlflow.end_run()
        mlflow.start_run(self.run_id)
        mlflow.log_params(self.config["TRAIN"])
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def _setup_dataloader(self):
        # Setup data loaders for training and validation
        if self.config["TRAIN"]["valid"]:
            train_images = find_images(os.path.join(self.train_dataset_dir, "images"))
            train_masks = find_images(os.path.join(self.train_dataset_dir, "masks"))
        else:
            train_images = find_images(
                [
                    os.path.join(self.train_dataset_dir, "images"),
                    os.path.join(self.val_dataset_dir, "images"),
                ]
            )
            train_masks = find_images(
                [
                    os.path.join(self.train_dataset_dir, "masks"),
                    os.path.join(self.val_dataset_dir, "masks"),
                ]
            )
        val_images = find_images(os.path.join(self.val_dataset_dir, "images"))
        val_masks = find_images(os.path.join(self.val_dataset_dir, "masks"))

        transform = get_augmentations()

        # Creating datasets for training and validation
        train_dataset = SegmentationDataset(
            image_paths=train_images,
            mask_paths=train_masks,
            num_classes=len(self.config["DATA"]["classes"]) + 1,
            channels=self.config["DATA"]["channels"],
            preprocess_image=preproc_image(
                self.config["DATA"]["img_size"], self.config["DATA"]["channels"]
            ),
            preprocess_mask=preproc_mask(self.config["DATA"]["img_size"]),
            transform=transform,
        )

        val_dataset = SegmentationDataset(
            image_paths=val_images,
            mask_paths=val_masks,
            num_classes=len(self.config["DATA"]["classes"]) + 1,
            channels=self.config["DATA"]["channels"],
            preprocess_image=preproc_image(
                self.config["DATA"]["img_size"], self.config["DATA"]["channels"]
            ),
            preprocess_mask=preproc_mask(self.config["DATA"]["img_size"]),
            transform=None,
        )

        num_workers = self.config["TRAIN"]["workers"]
        prs_workers = True if num_workers > 0 else False

        # DataLoader setup for training and validation datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["TRAIN"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=prs_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["TRAIN"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _setup_model(self):
        # Model setup and directory creation for model saving
        self.model_path = os.path.join("experiments", self.project_name, self.run_name)
        os.makedirs(self.model_path, exist_ok=True)

        self.model = DeepLabV3PlusModule(
            classes=self.config["DATA"]["classes"],
            encoder=self.config["TRAIN"]["encoder"],
            lr=self.config["TRAIN"]["learning_rate"],
        )

        if self.config["TRAIN"]["pretrained"]:
            self.model.load_from_checkpoint(
                encoder=self.config["TRAIN"]["encoder"],
                checkpoint_path=self.config["TRAIN"]["pretrained"],
                classes=self.config["DATA"]["classes"],
                strict=False,
            )

    def _setup_callbacks(self):
        # Setting up various training callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        save_period = (
            None
            if self.config["TRAIN"]["save_period"] < 1
            else self.config["TRAIN"]["save_period"]
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            save_top_k=1,
            save_last=True,
            monitor="val_iou",
            mode="max",
            save_weights_only=True,
            every_n_epochs=save_period,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=30, verbose=False, mode="min"
        )

        post_train_callback = PostTrainCallback(
            self.model_path, self.config, self.registered_model_name
        )

        self.callbacks = [
            checkpoint_callback,
            lr_monitor,
            post_train_callback,
            early_stop_callback,
        ]

    def _setup_logger(self):

        # Setup for logging with CSV
        csv_logger = CSVLogger(save_dir=self.model_path, name="training_log")

        mlflow_logger = MLFlowLogger(
            run_id=self.run_id,
        )

        self.run_id = mlflow_logger.run_id
        self.loggers = [mlflow_logger, csv_logger]

    def _setup_trainer(self):
        # Setting up the PyTorch Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config["TRAIN"]["epochs"],
            accelerator="gpu",
            devices=self.config["TRAIN"]["gpus"],
            callbacks=self.callbacks,
            log_every_n_steps=1,
            logger=self.loggers,
        )

    def __draw_results(self):
        # Drawing results from validation process
        best_model_path = self.trainer.checkpoint_callback.best_model_path

        model = DeepLabV3PlusModule.load_from_checkpoint(
            encoder=self.config["TRAIN"]["encoder"],
            checkpoint_path=best_model_path,
            classes=self.config["DATA"]["classes"],
            strict=False,
        )
        model.eval()
        device = torch.device("cuda" if model.on_gpu else "cpu")
        val_images = find_images(os.path.join(self.val_dataset_dir, "images"))
        preproc = preproc_image(
            self.config["DATA"]["img_size"], self.config["DATA"]["channels"]
        )

        save_dir = os.path.join(self.model_path, "valid_results")
        os.makedirs(save_dir, exist_ok=True)

        print("Drawing results...")
        for image_path in val_images:
            image0 = cv2.imread(image_path)
            image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image = preproc(image=image)["image"]
            image = torch.unsqueeze(image, 0)
            image = image.to(device)

            with torch.no_grad():
                output = model(image)
                output = torch.argmax(output, dim=1)
                output = output.cpu().numpy()

            output = np.squeeze(output)
            output = output.astype(np.float32)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

            colors_idx = np.unique(output)
            colors_idx = colors_idx[colors_idx != 0]

            if len(colors_idx) > 0:
                for idx in colors_idx:
                    output[output[:, :, 0] == idx] = COLOR_PALLET[int(idx) - 1]

                output = output.astype(np.uint8)
                output = cv2.resize(
                    output,
                    (image0.shape[1], image0.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                masked_image = cv2.addWeighted(image0, 1.0, output, 0.7, 0)
                cv2.imwrite(
                    os.path.join(save_dir, os.path.basename(image_path)), masked_image
                )
            else:
                cv2.imwrite(
                    os.path.join(save_dir, os.path.basename(image_path)), image0
                )

        # TODO: redo with logging
        print("Results drawn successfully!")

    def _get_project_and_run_name(self) -> Tuple[str]:
        return get_project_and_run_name("deeplabv3")
