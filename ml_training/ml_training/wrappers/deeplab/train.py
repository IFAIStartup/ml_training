from ml_training.wrappers.deeplab.wrapper import DeepLabTrainWrapper
from ml_training.wrappers.deeplab.tools import find_images, get_config


def train():
    config = get_config()
    wrapper = DeepLabTrainWrapper(config)
    wrapper.run()


if __name__ == '__main__':
    train()
