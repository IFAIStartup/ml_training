from ml_training.wrappers.deeplab.wrapper import DeepLabTrainWrapper
from ml_training.wrappers.deeplab.tools import find_images, get_config


def train():
    config = get_config()
    registered_model_name = "satellite_deeplabv3"
    wrapper = DeepLabTrainWrapper(config, registered_model_name)
    wrapper.train()


if __name__ == '__main__':
    train()
