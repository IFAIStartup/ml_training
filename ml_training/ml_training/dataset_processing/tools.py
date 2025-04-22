import os
import cv2
import glob
import numpy as np
import json
import datetime
import rasterio
import zipfile
from pathlib import Path
from filesplit.split import Split
from shapely import Polygon, union_all

from ml_training.dataset_processing.annotation import Annotation, AnnotatedObject


def merge_and_split(annot: Annotation, category_name: str):
    category_id = annot.categories.index(category_name)

    for img_name in annot.images:
        img = annot.images[img_name]

        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        new_objs = []
        for obj in img.annotations:
            if obj.category_id != category_id:
                new_objs.append(obj)
                continue

            for segment in obj.segmentation:
                segment = np.array(segment)
                segment = segment.astype('int32')
                segment = segment.reshape(-1, 1, 2)
                cv2.fillPoly(mask, [segment], 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            segment = cnt.reshape(-1).tolist()
            obj = AnnotatedObject([x, y, w, h], category_id, [segment])
            new_objs.append(obj)
        
        img.annotations = new_objs
    return annot



def get_dataset_path(datasets_dir: str, base_name: str):
    now = datetime.datetime.now()
    strf_date = now.strftime('%d%m%Y')
    name_with_date = f"{base_name}_{strf_date}"
    version_num = len(glob.glob(os.path.join(datasets_dir, name_with_date + '*')))
    name = f"{name_with_date}__v_{version_num}"
    
    return os.path.join(datasets_dir, name)


def change_annotation(annot: Annotation, new_classes: list):
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
    

def delete_small_bboxes(annot: Annotation):
    images = annot.images
    for name in images:
        new_bboxes = []
        
        for bbox in images[name].annotations:
            if annot.categories[bbox.category_id] != 'household_garbage':
                continue
        
            x, y, w, h = bbox['bbox']
            if w > 20 and h > 20:
                bbox.category_id = 0
                new_bboxes.append(bbox)
        images[name].annotations = new_bboxes
        
    annot.categories = ['household_garbage']
    return annot
    

def create_splitted_dataset(src_dir: str, dst_dir: str, block_volume: int):
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for sample_name in ['train', 'valid']:
        sample_path = os.path.join(src_dir, sample_name)
        
        with zipfile.ZipFile(f"{sample_path}.zip", mode="w") as archive:
            directory = Path(sample_path)
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))

        split = Split(f"{sample_path}.zip", dst_dir)
        split.bysize(block_volume)


def read_tiff(path):
    dataset = rasterio.open(path)

    bands = []
    for i in [3, 2, 1]:
        band = dataset.read(i)
        bands.append(band)
        
    img = cv2.merge(bands)
    return img
