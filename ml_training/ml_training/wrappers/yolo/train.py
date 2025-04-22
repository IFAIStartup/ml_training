from pathlib import Path
from ml_training.wrappers.yolo.tools import get_config
from ml_training.wrappers.yolo.wrapper import YoloTrainWrapper


def main():
    config = get_config()
    wrapper = YoloTrainWrapper(config)
    wrapper.train()
    wrapper.onnx_export()
    wrapper.create_pbtxt()


if __name__ == '__main__':
    main()
