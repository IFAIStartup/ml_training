from __future__ import annotations
import os
import sys
import cv2
import copy
import random
import math
import numpy as np
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum
import logging
import time
import shutil

from ml_training.dataset_processing.annotation import (
    Annotation,
    write_yolo_det,
    write_yolo_iseg,
    write_coco,
)
from ml_training.dataset_processing.image_source import (
    ImageSource,
    Resizer,
    Renamer,
    paths2image_sources,
)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Dataset:

    image_sources: List[ImageSource]
    annotation: Annotation
    subsets: Dict[str, int]

    def __init__(
        self,
        image_sources: List[ImageSource] = None,
        annotation: Annotation = None,
        subsets: Dict[str, List[int]] = None,
    ):
        """Constructor

        :param image_sources: list image sources, representing images, that will be placed in dataset, defaults to None
        :param annotation: annotation of images, represented by image sources, defaults to None
        :param samples: dict of lists of indexes of images, that corresponds to a specific set, defaults to None
        """

        self.image_sources = image_sources or []
        self.annotation = annotation or Annotation()
        self.subsets = subsets or {}

    def __len__(self):
        return len(self.image_sources)

    def __getitem__(self, item):
        return self.image_sources[item]

    def __add__(self, other: Dataset):

        # Addition of image sources
        sum_image_sources = self.image_sources + other.image_sources

        # Addition of annotation
        # sum_annotation = self.annotation
        # self.annotation['images'].update(other.annotation['images'])
        sum_annotation = self.annotation + other.annotation

        # Addition of susets
        self_sample_names = set(self.subsets.keys())
        other_sample_names = set(other.subsets.keys())

        # sum_sample_names - union of two sample names
        sum_sample_names = self_sample_names or other_sample_names
        sum_samples = {}

        # In new samples self indexes remain their values, others - are addicted with number of images in self
        # (other images addict to the end of common list)
        for name in sum_sample_names:
            sum_samples[name] = []
            if name in self_sample_names:
                sum_samples[name] += self.subsets[name]
            if name in other_sample_names:
                sum_samples[name] += list(
                    map(lambda x: x + len(self), other.subsets[name])
                )

        return Dataset(sum_image_sources, sum_annotation, sum_samples)

    def resize(self, size: tuple):
        assert len(size) == 2

        # Add resize fn to image sources
        for img_src in self.image_sources:
            img_src.editors.append(Resizer(size))

        # Go through annotation and correct coordinates
        new_width, new_height = size
        for image_name in self.annotation.images:
            labeled_image = self.annotation.images[image_name]

            old_width = labeled_image.width
            old_height = labeled_image.height

            # Correct image size
            labeled_image.width = new_width
            labeled_image.height = new_height

            # Correct bbox coordinates of cur image
            for bbox in labeled_image.annotations:
                x, y, w, h = bbox.bbox

                x *= new_width / old_width
                w *= new_width / old_width
                y *= new_height / old_height
                h *= new_height / old_height

                bbox.bbox = [x, y, w, h]

                segmentation = bbox.segmentation
                for i, segment in enumerate(segmentation):
                    segment = np.array(segment).astype("float64").reshape(-1, 1, 2)
                    segment[..., 0] *= new_width / old_width
                    segment[..., 1] *= new_height / old_height
                    segmentation[i] = segment.reshape(-1).astype("int32").tolist()
                    pass

    def rename(self, rename_callback: Callable):

        for i in range(len(self.image_sources)):

            # Rename image sources
            # old_name = self.image_sources[i].name
            # new_name = rename_callback(old_name)
            # self.image_sources[i].name = new_name
            old_name = self.image_sources[i].get_final_name()
            new_name = rename_callback(old_name)
            renamer = Renamer(rename_callback)
            self.image_sources[i].editors.append(renamer)

            # Rename annotations
            if old_name in self.annotation.images:
                image_info = self.annotation.images[old_name]
                self.annotation.images.pop(old_name)
                self.annotation.images[new_name] = image_info

    def split_by_proportions(self, proportions: dict):
        all_idx = [i for i in range(len(self.image_sources))]
        random.shuffle(all_idx)

        length = len(self.image_sources)
        split_start_idx = 0
        split_end_idx = 0

        # Reset current split indexes
        self.subsets = {}

        num_of_names = len(proportions.keys())

        for i, split_name in enumerate(proportions.keys()):
            split_end_idx += math.ceil(proportions[split_name] * length)
            self.subsets[split_name] = all_idx[split_start_idx:split_end_idx]
            split_start_idx = split_end_idx

            if i + 1 == num_of_names and split_end_idx < len(all_idx):
                self.subsets[split_name] += all_idx[split_end_idx : len(all_idx)]

        # logging
        message = "In dataset the following splits was created: "
        for i, split_name in enumerate(self.subsets.keys()):
            message += f"{split_name}({len(self.subsets[split_name])})"
            if i != len(self.subsets.keys()) - 1:
                message += ", "
        LOGGER.info(message)

    def split_by_dataset(self, yolo_dataset_path: str):

        # Define names of splits as dirnames in dataset directory
        split_names = [
            name
            for name in os.listdir(yolo_dataset_path)
            if os.path.isdir(os.path.join(yolo_dataset_path, name))
        ]

        # Reset current split indexes
        self.subsets = {}

        for split_name in split_names:

            # Place names of orig dataset split in set structure
            orig_dataset_files = os.listdir(
                os.path.join(yolo_dataset_path, split_name, "labels")
            )
            orig_names_set = set()

            for file in orig_dataset_files:
                name, ext = os.path.splitext(file)
                orig_names_set.add(name)

            # If new_name in orig dataset split then update split indexes of current dataset
            self.subsets[split_name] = []
            for i, image_source in enumerate(self.image_sources):
                new_name = image_source.get_final_name()
                if new_name in orig_names_set:
                    self.subsets[split_name].append(i)

    def add_with_proportion(self, dataset, proportions: dict):

        assert proportions.keys() == self.subsets.keys()

        orig_length = len(self)
        dataset_length = len(dataset)
        result_length = orig_length + dataset_length

        dataset_proportions = {}
        for name in self.subsets:
            orig_sample_length = len(self.subsets[name])
            result_sample_length = proportions[name] * result_length
            dataset_proportions[name] = (
                result_sample_length - orig_sample_length
            ) / dataset_length

        dataset.split_by_proportions(dataset_proportions)
        new_dataset = self + dataset

        # logging
        message = "Create summary dataset with samples: "
        for i, split_name in enumerate(self.subsets.keys()):
            message += f"{split_name}({len(self.subsets[split_name])})"
            if i != len(self.subsets.keys()) - 1:
                message += ", "
        LOGGER.info(message)

        return new_dataset

    def remove_empty_images(self, residual_empty_percentage: float = 0):
        assert 0 <= residual_empty_percentage < 1

        empty_img_srcs = []
        filled_img_srcs = []

        for img_src in self.image_sources:
            name = img_src.get_final_name()
            if name not in self.annotation.images:
                empty_img_srcs.append(img_src)
                continue

            bboxes = self.annotation.images[name].annotations
            if len(bboxes) == 0:
                empty_img_srcs.append(img_src)
                continue

            # # TODO: CHECK
            # bboxes_is_empty = True
            # for bbox in bboxes:
            #     if len(bbox.segmentation) != 0:
            #         bboxes_is_empty = False
            #         break
            # if bboxes_is_empty:
            #     continue

            filled_img_srcs.append(img_src)

        random.shuffle(empty_img_srcs)
        residual_empty_img_srcs = empty_img_srcs[
            0 : int(
                len(filled_img_srcs)
                * residual_empty_percentage
                / (1 - residual_empty_percentage)
            )
        ]
        self.image_sources = filled_img_srcs + residual_empty_img_srcs

    def install(
        self,
        dataset_path: str,
        dataset_name: str = "dataset",
        image_ext: str = ".jpg",
        install_images: bool = True,
        install_yolo_det_labels: bool = False,
        install_yolo_seg_labels: bool = False,
        install_coco_annotations: bool = True,
        install_masks: bool = False,
        install_description: bool = True,
    ):

        assert (install_yolo_det_labels and install_yolo_seg_labels) == False

        for subset_name in self.subsets.keys():
            if install_images:
                self._install_images(dataset_path, subset_name, image_ext)

            subset_annotation = self._get_subset_annotation(subset_name)
            if install_yolo_det_labels:
                self._install_yolo_det_labels(
                    subset_annotation, dataset_path, subset_name
                )

            if install_yolo_seg_labels:
                self._install_yolo_seg_labels(
                    subset_annotation, dataset_path, subset_name
                )

            if install_coco_annotations:
                self._install_coco_annotations(
                    subset_annotation, dataset_path, subset_name, image_ext
                )

            if install_masks:
                self._install_masks(subset_annotation, dataset_path, subset_name)

        if install_description:
            self._write_description(
                os.path.join(dataset_path, "data.yaml"), dataset_name
            )

        self._clear_cache(dataset_path)

    def _install_images(self, dataset_path, subset_name, image_ext):
        subset_ids = self.subsets[subset_name]
        images_dir = os.path.join(dataset_path, subset_name, "images")
        os.makedirs(images_dir, exist_ok=True)

        for i, split_idx in enumerate(subset_ids):
            image_source = self.image_sources[split_idx]
            # save_img_path = os.path.join(images_dir, image_source.name + image_ext)
            image_source.save(
                images_dir,
                image_ext,
                cache_dir=os.path.join(dataset_path, ".cvml2_cache"),
            )

            LOGGER.info(
                f"[{i + 1}/{len(subset_ids)}] "
                + f"{subset_name}:{self.image_sources[split_idx].get_final_name()}{image_ext} is done"
            )
        LOGGER.info(f"{subset_name} is done")

    def _install_yolo_det_labels(
        self, subset_annotation: Annotation, dataset_path: str, subset_name: str
    ):

        labels_dir = os.path.join(dataset_path, subset_name, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        write_yolo_det(subset_annotation, labels_dir)
        LOGGER.info(f"{subset_name}:yolo_labels is done")

    def _install_yolo_seg_labels(
        self, subset_annotation: Annotation, dataset_path: str, subset_name: str
    ):

        labels_dir = os.path.join(dataset_path, subset_name, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        write_yolo_iseg(subset_annotation, labels_dir)
        LOGGER.info(f"{subset_name}:yolo_labels is done")

    def _install_coco_annotations(
        self,
        subset_annotation: Annotation,
        dataset_path: str,
        subset_name: str,
        image_ext: str,
    ):

        annotation_dir = os.path.join(dataset_path, subset_name, "annotations")
        os.makedirs(annotation_dir, exist_ok=True)
        coco_path = os.path.join(annotation_dir, "data.json")
        write_coco(subset_annotation, coco_path, image_ext)
        LOGGER.info(f"{subset_name}:coco_annotation is done")

    def _install_masks(
        self, subset_annotation: Annotation, dataset_path: str, subset_name: str
    ):

        masks_dir = os.path.join(dataset_path, subset_name, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        colors = self._get_segment_colors(subset_annotation.categories)

        for name, annot_image in subset_annotation.images.items():
            width = annot_image.width
            height = annot_image.height
            mask = np.zeros((height, width), dtype="uint8")

            for bbox in annot_image.annotations:
                cat_id = bbox.category_id
                segmentation = bbox.segmentation

                for segment in segmentation:
                    segment = np.array(segment)
                    segment = segment.reshape((-1, 1, 2))
                    segment = segment.astype("int32")

                    color = colors[cat_id]
                    cv2.fillPoly(mask, [segment], color)

            mask_path = os.path.join(masks_dir, name + ".png")
            cv2.imwrite(mask_path, mask)

    # TODO: number of colors is limited
    def _get_segment_colors(self, class_names: list):
        colors = []
        max_color = 255
        for i, class_name in enumerate(class_names):
            color = (max_color * (i + 1)) // len(class_names)
            colors.append(color)

        return colors

    def _get_subset_annotation(self, sample_name: str) -> Annotation:
        sample_classes = self.annotation.categories
        sample_images = {}

        for i in self.subsets[sample_name]:
            name = self.image_sources[i].get_final_name()
            if name not in self.annotation.images:
                continue
            sample_images[name] = self.annotation.images[name]

        sample_annotation = Annotation(categories=sample_classes, images=sample_images)
        return sample_annotation

    def _write_description(self, path: str, dataset_name: str):
        text = (
            f"train: {dataset_name}/train/images\n"
            f"val: {dataset_name}/valid/images\n\n"
            f"nc: {len(self.annotation.categories)}\n"
            f"names: {self.annotation.categories}"
        )
        with open(path, "w") as f:
            f.write(text)
        LOGGER.info(f"Description is done")

    def _clear_cache(self, dataset_path):
        if os.path.exists(os.path.join(dataset_path, ".cvml2_cache")):
            shutil.rmtree(os.path.join(dataset_path, ".cvml2_cache"))
