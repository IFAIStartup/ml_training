#!/bin/bash

docker stop ml_traning-web-1
sleep 5
docker rm ml_traning-web-1
sleep 2

docker stop  ml_traning-worker-1
sleep 5
docker rm  ml_traning-worker-1
sleep 2

docker stop ml_traning-triton_server-1
sleep 5
docker rm ml_traning-triton_server-1
sleep 2

docker stop ml_traning-mlflow-1
sleep 5
docker rm ml_traning-mlflow-1
sleep 2

docker stop ml_traning-rabbitmq-1
sleep 5
docker rm ml_traning-rabbitmq-1
sleep 2

docker compose up -d