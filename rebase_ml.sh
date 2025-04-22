#!/bin/bash

docker stop ml_traning-web-1
docker rm ml_traning-web-1
docker rmi ml_traning-web

docker stop  ml_traning-worker-1
docker rm  ml_traning-worker-1
docker rmi ml_traning-worker 

docker stop ml_traning-triton_server-1
docker rm ml_traning-triton_server-1

# when rebase this container all uncomments
# docker stop ml_traning-mlflow-1
# docker rm ml_traning-mlflow-1
# docker rmi ml_traning-mlflow

docker stop ml_traning-rabbitmq-1
docker rm ml_traning-rabbitmq-1

# echo "sudo password"
# sudo rm -rf ml_training/experiments/*
# sudo rm -rf ml_training/static/*

docker compose up -d