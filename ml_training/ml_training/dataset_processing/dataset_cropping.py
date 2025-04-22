import cv2
import os
import sys
import copy
import numpy as np
from typing import Tuple
import albumentations as A
from ml_training.dataset_processing import Dataset, Annotation, AnnotatedImage, AnnotatedObject
from ml_training.dataset_processing.image_source import ImageSource, ImageEditor
from shapely import Polygon, MultiPolygon, intersection, union_all


class Cropper(ImageEditor):
    def __init__(self, size: Tuple[int, int], idx: int, overlap: int = 0) -> None:
        super().__init__()
        self.size = size
        self.idx = idx
        self.overlap = overlap
    
    def edit_name(self, name: str) -> str:
        return self._edit_name_by_index(name, self.idx)
    
    def edit_image(self, img: np.ndarray, name: str) -> Tuple[np.ndarray, dict or None]:        
        crops = split_image(img, self.size, self.overlap)

        cache = {}
        for i, crop in enumerate(crops):
            cache[self._edit_name_by_index(name, i)] = crop
        
        return crops[self.idx], cache
        
    def _edit_name_by_index(self, name: str, idx: int):
        return name + '_' + str(idx)


def split_image(img: np.ndarray, size: tuple, overlap: int = 0) -> list:
    """
    Split image into tiles with overlap.
    :param image: image to split
    :param tile_size: size of tile
    :param overlap: overlap between tiles
    :return: list of tiles
    """
    height, width = img.shape[:2]
    h, w = size[:2]
    
    num_rows, num_cols = get_num_rows_cols(width, height, w, h, overlap)

    padded_height = num_rows * (h - overlap) + overlap
    padded_width = num_cols * (w - overlap) + overlap
    img_pad = add_padding(img, (padded_height, padded_width))

    tiles = []
    for row in range(num_rows):
        for col in range(num_cols):
            
            x1 = col * (w - overlap)
            y1 = row * (h - overlap)
            x2 = x1 + w
            y2 = y1 + h
            
            tile = img_pad[y1:y2, x1:x2]
            tiles.append(tile)
            
    return tiles


def get_num_rows_cols(img_width, img_height, crop_w, crop_h, overlap):
    num_rows = img_height // (crop_h - overlap)
    num_cols = img_width // (crop_w - overlap)

    if num_rows * (crop_h - overlap) + overlap < img_height:
        num_rows += 1

    if num_cols * (crop_w - overlap) + overlap < img_width:
        num_cols += 1
        
    return num_rows, num_cols


def add_padding(image: np.ndarray, size: (int, int)) -> np.ndarray:
    """
    Add padding to image
    :param image:
    :param size:
    :return: image with padding
    """
    transform = A.Compose([
        A.PadIfNeeded(min_height=size[0], min_width=size[1], border_mode=cv2.BORDER_CONSTANT, value=0)
    ])
    image = transform(image=image)['image']
    return image


def crop_dataset(dataset: Dataset, size: tuple, overlap = 0) -> Dataset:
    image_sources = dataset.image_sources
    annotation = dataset.annotation
    categories = annotation.categories
    
    new_image_sources = []
    new_annotation = Annotation(categories=categories, images={})
    
    for img_src in image_sources:
        name = img_src.get_final_name()
        if name not in annotation.images:
            continue
        
        lbl_image = annotation.images[name]
        cur_new_img_srcs, cur_new_lbl_images = crop_dataset_image(img_src, lbl_image, size, overlap)

        new_image_sources += cur_new_img_srcs
        new_annotation.images.update(cur_new_lbl_images)
    
    new_dataset = Dataset(new_image_sources, new_annotation)
    return new_dataset
    
    
# TODO: subsets
def crop_dataset_image(image_source: ImageSource, labeled_image: AnnotatedImage, crop_size: tuple, overlap: int):
    new_img_srcs, new_lbl_images = [], {}
    width, height = labeled_image.width, labeled_image.height
    
    crop_w, crop_h = crop_size
    
    num_rows, num_cols = get_num_rows_cols(width, height, crop_w, crop_h, overlap)
    cnt = 0
    
    name = image_source.get_final_name()
    
    padded_height = num_rows * (crop_h - overlap) + overlap
    padded_width = num_cols * (crop_w - overlap) + overlap
    
    left_pad = (padded_width - width) // 2
    upper_pad = (padded_height - height) // 2
    
    for row in range(num_rows):
        for col in range(num_cols):
            
            x1 = col * (crop_w - overlap) - left_pad
            y1 = row * (crop_h - overlap) - upper_pad
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            new_img_src = ImageSource(
                image_source.path, 
                image_source.name,
                image_source.editors[:]
            )
            
            cropper = Cropper(crop_size, cnt, overlap)
            new_img_src.editors.append(cropper)
            new_img_srcs.append(new_img_src)
            
            new_bboxes = []
            for bbox in labeled_image.annotations:
                old_x, old_y, old_w, old_h = bbox.bbox
                if old_x >= x2 or old_y >= y2:
                    continue
                if old_x + old_w <= x1 or old_y + old_h <= y1:
                    continue
                
                new_x = max(0, old_x - x1)
                new_y = max(0, old_y - y1)
                new_x2 = min(crop_w, new_x + old_w)
                new_y2 = min(crop_h, new_y + old_h)
                new_w = new_x2 - new_x
                new_h = new_y2 - new_y
                
                segmentation = bbox.segmentation
                if len(segmentation) != 0:
                    bbox_is_with_segmentation = True
                    new_segmentation = crop_segmentation(
                        segmentation, 
                        (x1, y1, x2, y2))
                else:
                    bbox_is_with_segmentation = False
                    new_segmentation = segmentation
                
                #TODO: check if segmentation is empty (we need to exlude bbox)
                if bbox_is_with_segmentation and len(new_segmentation) == 0:
                    continue

                new_bbox = {
                    'category_id': bbox.category_id,
                    'bbox': [new_x, new_y, new_w, new_h],
                    'segmentation': new_segmentation,
                }
                new_bbox = AnnotatedObject(**new_bbox)
                new_bboxes.append(new_bbox)
            
            new_lbl_image = AnnotatedImage(width=crop_w, height=crop_h, annotations=new_bboxes)
            new_lbl_images[new_img_src.get_final_name()] = new_lbl_image
            
            cnt += 1
    return new_img_srcs, new_lbl_images
        

# def crop_segmentation(segmentation, xyxy):
#     x1, y1, x2, y2 = xyxy
#     mask = np.zeros((y2 - y1, x2 - x1), dtype='uint8')
        
#     for segment in segmentation:
#         segment = np.array(segment)
        
#         segment = segment.astype('int32')
#         segment = segment.reshape(-1, 1, 2)
        
#         segment[..., 0] -= x1
#         segment[..., 1] -= y1
        
#         cv2.fillPoly(mask, [segment], 255)
    
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     new_segmentation = []
#     for cnt in contours:
#         cnt = cnt.reshape(-1)
#         new_segmentation.append(cnt.tolist())
    
#     return new_segmentation


def crop_segmentation(segmentation, xyxy):
    x1, y1, x2, y2 = xyxy
    img_rect = Polygon(((0, 0), (0, y2 - y1 - 1), 
                        (x2 - x1 - 1, y2 - y1 - 1), (x2 - x1 - 1, 0)))
    
    obj_polys = []
    new_segmentation = []
    for segment in segmentation:
        segment = np.array(segment)
        
        segment = segment.reshape(-1, 2)

        if len(segment) < 3:
            continue
        
        segment[..., 0] -= x1
        segment[..., 1] -= y1

        segment = segment.tolist()
        segment.append(segment[0])
        
        obj_poly = Polygon(segment)
        # obj_polys.append(obj_poly)

        if not obj_poly.is_valid:
            obj_poly = obj_poly.buffer(0)

        new_obj_poly = intersection(obj_poly, img_rect)
    
        if type(new_obj_poly) == Polygon:
            new_obj_poly = MultiPolygon([new_obj_poly])
        elif type(new_obj_poly) == MultiPolygon:
            pass
        else:
            continue
        
        for poly in new_obj_poly.geoms:
            xs, ys = poly.exterior.coords.xy
            xs, ys = xs.tolist(), ys.tolist()
            
            new_segment = np.array([xs, ys]).T.reshape(-1).tolist()[:-2]
            new_segmentation.append(new_segment)
    
    return new_segmentation
        
