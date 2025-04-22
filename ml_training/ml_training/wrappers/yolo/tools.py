import os
import yaml

DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), 'args.yaml')


def get_config(config_path: str = DEFAULT_CFG_PATH):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
