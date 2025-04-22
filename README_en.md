
# ML Training

## Description

This repository contains code for training neural network models and deploying trained models using MLFlow, Celery, and Triton Server.

## Components

- FastAPI: Used for interacting with the application.
- Celery Worker: A Celery Worker is launched via RabbitMQ message broker for training.
- MLFlow Server: MLFlow Server is used to log metrics and models.
- Triton Server: Triton Server is launched to deploy trained models.

## Installation and Usage

Follow these steps to set up and run your neural network training project:

1. Data Preparation:

- By default, the `./datasets` folder is mounted into the container and is used as the storage for input datasets. Place a [sample dataset](https://disk.yandex.ru/d/1EqFcYBCHdg_AQ) there to enable training. The format of the input datasets is described below.

2. In the root folder of the project, run:
```
docker-compose up
```
This will start all necessary containers.

3. Interacting with the Application:
- Navigate to http://localhost:8181/docs# in your web browser.
- Through the Swagger interface, you can make requests and test your application’s functionality.

4. Starting Training:
- Send a request to the `/train/` endpoint via FastAPI. The default parameters will start training of two runs: yolov8 and deeplabv3.
- The Celery worker will begin processing the input dataset (in COCO format) and start the training process.
Metrics and results will be logged in MLFlow.

5. Logging in MLFlow:
- Go to http://localhost:5000/ to access the MLFlow Server and view training logs. The admin login and password are set in `basic_auth.ini`.

## Input Dataset Format

Your input dataset should be organized as follows:

- `images` folder: This folder should contain the images you want to use for model training. Images can be in JPEG, PNG, or other supported formats. File names can be arbitrary, but it is recommended to follow a naming system (e.g., by numbers or object names in the image).
- `annotations` folder: This folder should contain the file `instances_default.json`. This file includes annotations of objects in the images in the COCO (Common Objects in Context) format.
