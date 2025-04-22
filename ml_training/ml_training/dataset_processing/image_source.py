import os
import cv2
import glob
import shutil
from abc import ABC
from typing import Any, List, Callable, Tuple
import numpy as np


class ImageEditor(ABC):
    def edit_name(self, name: str) -> str:
        return name
    
    def edit_image(self, img: np.ndarray, name: str) -> Tuple[np.ndarray]:
        return img, None


class FunctionImageEditor(ImageEditor):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func
    
    def edit_image(self, 
                   img: np.ndarray, 
                   name: str,) -> Tuple[np.ndarray, dict or None]:
        return self.func(img), None


class Resizer(ImageEditor):
    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self.size = size
    
    def edit_image(self, 
                   img: np.ndarray, 
                   name: str) -> Tuple[np.ndarray, dict or None]:
        img = cv2.resize(img, self.size)
        return img, None

class Renamer(ImageEditor):
    def __init__(self, rename_callback: Callable) -> None:
        super().__init__()
        self.rename_callback = rename_callback
        
    def edit_name(self, name: str) -> str:
        return self.rename_callback(name)


class ImageSource:
    name: str
    editors: List[ImageEditor]
    
    def __init__(self, 
                 path: str,
                 name: str = None, 
                 editors: List[ImageEditor] = None) -> None:
        
        self.path = path
        self.editors = editors or []
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        
    def save(self, save_dir: str, image_ext: str = '.jpg', cache_dir = None):
        final_name = self.get_final_name()
        save_path = os.path.join(save_dir, final_name + image_ext)
        
        ret = self._try_to_save_cached(cache_dir, final_name, image_ext, save_path)
        if ret is True:
            return
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        img = self.get_final_img(cache_dir, image_ext)
        self._write(save_path, img)

    def get_final_name(self):
        name = self.name
        for editor in self.editors:
            name = editor.edit_name(name)
        return name
    
    def get_final_img(self, cache_dir: str = None, image_ext='.jpg'):
        edit_idx, name, img = self._check_cache_or_read(cache_dir)    
        
        for i, editor in enumerate(self.editors[edit_idx + 1:]):
            img, img_cache = editor.edit_image(img, name)
            self._save_img_cache(i, len(self.editors), img_cache, cache_dir, image_ext)
            name = editor.edit_name(name)
            
        return img
    
    def _try_to_save_cached(self, cache_dir, final_name, image_ext, save_path):
        if cache_dir is not None:
            cached_file = os.path.join(cache_dir, final_name + image_ext)
            if os.path.exists(cached_file):
                shutil.copy(cached_file, save_path)
                return True
        return False
    
    def _check_cache_or_read(self, cache_dir):
        if cache_dir is None:
            return -1, self.name, self._read(self.path)
        
        name = self.name
        names = []
        for editor in self.editors:
            name = editor.edit_name(name)
            names.append(name)
        
        for i, n in enumerate(names[::-1]):
            if i == 0:
                continue
            
            cached_path = os.path.join(cache_dir, n + '.npy')
            if os.path.exists(cached_path):
                img = np.load(cached_path)
                return len(names) - 1 - i, n, img
        
        return -1, self.name, self._read(self.path)
    
    def _save_img_cache(self, i, num_editors, img_cache, cache_dir, image_ext):
        if cache_dir is None or img_cache is None:
            return 
        
        for name in img_cache:
            ext = image_ext if i == num_editors - 1 else '.npy'
            path = os.path.join(cache_dir, name + ext)
        
            if i == num_editors - 1:
                self._write(path, img_cache[name])
            else:
                np.save(path, img_cache[name])
                
        
    def _read(self, path: str) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    
    def _write(self, path: str, img: np.ndarray):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        ext = os.path.splitext(os.path.split(path)[-1])[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        im_buf_arr.tofile(path)


# def crop_image(img: np.ndarray, size: tuple):        
#     width, height = img.shape[1], img.shape[0]
    
#     crop_w, crop_h = size
    
#     num_rows = -(-height // crop_h)
#     num_cols = -(-width // crop_w)
#     cnt = 0

#     crops = []    
#     for row in range(num_rows):
#         for col in range(num_cols):
            
#             x2 = min(width, crop_w * (col + 1))
#             y2 = min(height, crop_h * (row + 1))

#             x1 = x2 - crop_w
#             y1 = y2 - crop_h

#             crop = img[y1:y2, x1:x2]
#             crops.append(crop)
    
#     return crops


def paths2image_sources(paths: List[str]) -> List[ImageSource]:
    
    image_sources = []
    for path in paths:
        image_source = ImageSource(path)
        image_sources.append(image_source)

    return image_sources

# class ImageSource(ABC):
#     name: str
    
#     def read(self) -> np.ndarray:
#         pass
    
#     def save(self, save_dir: str, image_ext: str = '.jpg', cache_dir = CACHE_DIR):
#         pass

# class PathImageSource(ImageSource):
#     """Common image source that can be read and saved with specific processing"""
    
#     def __init__(self, path: str, preprocessing_fns: List[Callable] = None, name: str = None):
#         """
#         :param path: path to image
#         :param preprocessing_fns: list of preprocessing functions 
#                                   that can be applied while saving, defaults to None
#         :param name: image name (it can differ from file name), defaults to None (filename without ext)
#         """
#         self.path = path
#         self.preprocessing_fns = preprocessing_fns or []
#         self.name = name or os.path.splitext(os.path.basename(path))[0]
    
#     def read(self) -> np.ndarray:
#         img = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_COLOR)
#         for fn in self.preprocessing_fns:
#             img = fn(img)
#         return img
    
#     def _write(self, path: str, img: np.ndarray):
#         ext = os.path.splitext(os.path.split(path)[-1])[1]
#         is_success, im_buf_arr = cv2.imencode(ext, img)
#         im_buf_arr.tofile(path)
    
#     def save(self, save_dir: str, image_ext: str = '.jpg', cache_dir = None):
#         img = self.read()
#         self._write(os.path.join(save_dir, self.name + image_ext), img)


# class CropImageSource(ImageSource):
#     """Common image source that can be read and saved with specific processing"""
    
#     def __init__(self, 
#                  original_image_source: ImageSource,
#                  idx: int,
#                  cropper_fn: Callable,  
#                  name: str):
        
#         self.original_image_source = original_image_source
#         self.idx = idx
#         self.cropper_fn = cropper_fn
#         self.name = name
        
#         # """
#         # :param path: path to image
#         # :param preprocessing_fns: list of preprocessing functions 
#         #                           that can be applied while saving, defaults to None
#         # :param name: image name (it can differ from file name), defaults to None (filename without ext)
#         # """
#         # self.path = path
#         # self.cropper_fn = cropper_fn
#         # self.idx = idx
#         # self.preprocessing_fns = preprocessing_fns or []
#         # self.name = name or os.path.splitext(os.path.basename(path))[0]
    
#     def read(self) -> np.ndarray:
#         img = self.original_image_source.read()
#         return img
    
#     def _write(self, path: str, img: np.ndarray):
#         ext = os.path.splitext(os.path.split(path)[-1])[1]
#         is_success, im_buf_arr = cv2.imencode(ext, img)
#         im_buf_arr.tofile(path)
    
#     def save(self, save_dir: str, image_ext: str = '.jpg', cache_dir: str = None):
        
#         if cache_dir is None:
#             img = self.read()
#             crops = self.cropper_fn(img)
#             crop = crops[self.idx]
#             cv2.imwrite(os.path.join(cache_dir, self.name + image_ext), crop)
#             return
        
#         os.makedirs(cache_dir, exist_ok=True)
#         cached_file = os.path.join(cache_dir, self.name + image_ext)
        
#         if not os.path.exists(cached_file):
#             img = self.read()
#             crops = self.cropper_fn(img)
#             for i, crop in enumerate(crops):
#                 cv2.imwrite(os.path.join(cache_dir, '_'.join(self.name.split('_')[:-1]) + '_' + str(i) + image_ext), crop)
#         else:
#             pass
#         if os.path.exists(os.path.join(save_dir, self.name + image_ext)):
#             os.remove(os.path.join(save_dir, self.name + image_ext))
#         os.rename(cached_file, os.path.join(save_dir, self.name + image_ext))
        

# def paths2image_sources(paths: List[str], 
#                         preprocess_fns: List[Callable] = None) -> List[PathImageSource]:
    
#     image_sources = []
#     for path in paths:
#         image_source = PathImageSource(path, preprocess_fns)
#         image_sources.append(image_source)

#     return image_sources


