from pathlib import Path
from ml_training.wrappers.yolo.tools import get_config
from ml_training.wrappers.yolo.wrapper import YoloTrainWrapper


def main():
    config = get_config()
    registered_model_name = "satellite_yolov8"
    wrapper = YoloTrainWrapper(config, registered_model_name)
    wrapper.train()
    

if __name__ == '__main__':
    main()
