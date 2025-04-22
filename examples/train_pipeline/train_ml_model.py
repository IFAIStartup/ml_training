import os
import json
import mlflow
from ml_training.train_pipeline.fast import Item
from ml_training.train_pipeline.fast import start_training_run
from ml_training.train_pipeline.fast import train_ml_model


if __name__ == '__main__':

    item = Item(
        data_path='/home/student2/workspace/ml_traning/datasets/geoai_satellite_deeplab_24012024__v_2',
        ml_model_type='deeplabv3')

    if item.ml_model_type not in ['yolov8', 'deeplabv3']:
        print('aaaaaa')
        exit()

    run_id = start_training_run(item.ml_model_type)
    item_dict = json.loads(item.model_dump_json())

    result = train_ml_model(item_dict, run_id)