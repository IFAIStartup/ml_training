import os
import glob
import yaml
from natsort import natsorted

DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), 'cfg.yaml')


def get_config(config_path: str = DEFAULT_CFG_PATH):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_files(paths, patterns):
    all_files = []
    for path in paths:
        for pattern in patterns:
            path_pattern = os.path.join(path, pattern)
            files = glob.glob(path_pattern)
            files = natsorted(files)
            all_files.extend(files)
    return all_files


def find_images(paths):
    if isinstance(paths, str):
        paths = [paths]

    return find_files(paths, ['*.jpg', '*.jpeg', '*.png', '*.bmp'])
