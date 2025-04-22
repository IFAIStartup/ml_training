import os
import yaml
import mlflow
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ml_training.wrappers.yolo.wrapper import YoloTrainWrapper
from ml_training.wrappers.deeplab.wrapper import DeepLabTrainWrapper 
from ml_training.wrappers.yolo.tools import get_config as get_yolo_config
from ml_training.wrappers.deeplab.tools import get_config as get_deeplab_config


def train_specified_model(run_info: dict, run_id: str = None, static: str = "static"):
    """Start training run with logging to specified MLFlow run

    Args:
        run_info (dict): run information
        run_id (str, optional): id of MLFLow run. Defaults to None.
    """
    print('train_specified_model')
    if run_info['ml_model_type'] == 'yolov8':
        train_yolov8(run_info, run_id)
    
    elif run_info['ml_model_type'] == 'yolov8_det':
        train_yolov8_det(run_info, run_id)

    elif run_info['ml_model_type'] == 'deeplabv3':
        train_deeplabv3(run_info, run_id)
    
    else:
        pass


def train_yolov8(item: dict, run_id: str = None):
    """Train YOLOv8 ML model using MLFlow Tracking Server

    Args:
        item (dict): training parameters that was recieved as a body of HTTP request
        run_id (str, optional): Unique run id in MLFLow Server, that will be used for logging. 
                                Defaults to None (which mean dont use specified run and create a new one).
    """

    # Prepare configs for train wrapper
    config = get_yolo_config()

    if item['config'] is not None:
        config.update(item['config'])
    config['data'] = os.path.join(item['data_path'], 'preprocessed_dataset', 'data.yaml')
    config['imgsz'] = item['img_size']
    config['epochs'] = item['epochs']
    registered_model_name = item['registered_model_name']
    
    # Train, log and register ML model using special wrapper
    wrapper = YoloTrainWrapper(config, registered_model_name, run_id)
    wrapper.train()


def train_yolov8_det(item: dict, run_id: str = None):
    """Train YOLOv8 ML model for detection task using MLFlow Tracking Server

    Args:
        item (dict): training parameters that was recieved as a body of HTTP request
        run_id (str, optional): Unique run id in MLFLow Server, that will be used for logging. 
                                Defaults to None (which mean dont use specified run and create a new one).
    """
    print('train_yolov8_det')
    # Prepare configs for train wrapper
    config = get_yolo_config()

    # Add detection params to default config
    config['model'] = 'yolov8n.pt'
    config['task'] = 'detect'
    
    if item['config'] is not None:
        config.update(item['config'])
    config['data'] = os.path.join(item['data_path'], 'preprocessed_dataset', 'data.yaml')
    config['imgsz'] = item['img_size']
    config['epochs'] = item['epochs']
    registered_model_name = item['registered_model_name']
    
    # Train, log and register ML model using special wrapper
    wrapper = YoloTrainWrapper(config, registered_model_name, run_id)
    wrapper.train()


def train_deeplabv3(item: dict, run_id: str = None):
    """Train DeepLabv3+ ML model using MLFlow Tracking Server

    Args:
        item (dict): training parameters that was recieved as a body of HTTP request
        run_id (str, optional): Unique run id in MLFLow Server, that will be used for logging. 
                                Defaults to None (which mean dont use specified run and create a new one).
    """
    
    # Prepare configs for train wrapper
    config = get_deeplab_config()
    config['DATA']['dataset_dir'] = os.path.join(item['data_path'], 'preprocessed_dataset')
    config['DATA']['img_size'] = item['img_size']
    config['TRAIN']['epochs'] = item['epochs']
    registered_model_name = item['registered_model_name']
    
    with open(os.path.join(item['data_path'], 'preprocessed_dataset', 'data.yaml')) as f:
        data = yaml.safe_load(f)
    config['DATA']['classes'] = data['names']
    
    # Train, log and register ML model using special wrapper
    wrapper = DeepLabTrainWrapper(config, registered_model_name, run_id)
    wrapper.train()


def dummy_train():
    my_dummy_list = []
    for x in range(10000):
        x = x ** 3
        my_dummy_list.append(x)
    return my_dummy_list


def simple_train(creds: dict) -> None:
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))


    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    print('  '.join(classes[labels[j]] for j in range(4)))

    import torch.nn as nn
    import torch.nn.functional as F

    # PyTorch models inherit from torch.nn.Module
    class GarmentClassifier(nn.Module):
        def __init__(self):
            super(GarmentClassifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = GarmentClassifier()

    loss_fn = torch.nn.CrossEntropyLoss()

    # NB: Loss functions expect data in batches, so we're creating batches of 4
    # Represents the model's confidence in each of the 10 classes for a given input
    dummy_outputs = torch.rand(4, 10)
    # Represents the correct class among the 10 being tested
    dummy_labels = torch.tensor([1, 5, 3, 7])

    print(dummy_outputs)
    print(dummy_labels)

    loss = loss_fn(dummy_outputs, dummy_labels)
    print('Total loss for this batch: {}'.format(loss.item()))

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 10

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == '__main__':
    # simple_train(None)
    # run_id = start_training_run('deeplab')
    item_dict = {
        'data_path': "/home/student2/workspace/ml_traning/datasets/geoai_aerial_deeplab_26012024__v_2",
        'classes': None,
        'model_type': "deeplab",
        'registered_model_name': "default_model",
        'config': None
    }
    result = train_deeplabv3(item_dict)