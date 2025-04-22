import os
import sys
from pathlib import Path
import pytest
import json
import requests
import onnx
import mlflow
from mlflow.models import infer_signature
import torch
from torch import nn
import subprocess
from ml_training.deployment.deployment_client import TritonDeploymentClient

TEST_FILES_DIR = str(Path(__file__).parent.parent / 'test_files')


class TestTritonDeploy:

    def test_deploy_one_model(self):
        
        self.setup_aaaa()
        
        print("Register simple onnx model")
        #register_simple_onnx_model()            

        print("Create Deployment")
        target_uri = 'http://127.0.0.1:5000'
        triton_url = 'http://127.0.0.1:8000'
        client = TritonDeploymentClient(target_uri, self.model_repository, triton_url)
        client.create_deployment('model_onnx', "models:/registered_model_onnx/1")
    
        #self.teardown_aaa()
        
        response = requests.post(triton_url + '/v2/repository/index')
        triton_models = json.loads(response.text)
        
        assert len(triton_models) == 1

    def setup_aaaa(self):
        print('Start Triton Server')
        self.model_repository = os.path.join(TEST_FILES_DIR, 'model_repository')
        self.triton_container_id = start_triton_server(self.model_repository)
        
        print('Start MLFlow Server')
        #start_mlflow()

    def teardown_aaa(self):
        print("Stop Triton Server")
        stop_docker_container(self.triton_container_id)
        
        print("Kill MLFlow Server")
        #kill_mlflow()

    


def start_triton_server(model_repository: str) -> str:
    command = [
        'docker', 'run',
        '--gpus=all', '--name', 'triton_server', '--rm', '-d',
        '-p', '8000:8000', '-p', '8001:8001', '-p', '8002:8002',
        '-v', f'{model_repository}:/models', 
        'nvcr.io/nvidia/tritonserver:22.08-py3', 'tritonserver', 
        '--model-repository=/models', '--model-control-mode=explicit',
    ]
    #subprocess.run(command, encoding='utf-8')
    container_id = subprocess.check_output(command, encoding='utf-8')
    container_id = container_id[:-1]
    return container_id


def stop_docker_container(container_id: str):
    command = ['docker', 'stop', container_id]
    subprocess.check_output(command, encoding='utf-8')

def start_mlflow(host: str = '0.0.0.0', port: str = '5000') -> str:
    python_path = sys.executable
    command = f'{python_path} -m mlflow server --host {host} --port {port}'.split()
    out = subprocess.Popen(command, shell=True)
    return out

def kill_mlflow():
    command = "ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9"
    os.system(command)

def register_simple_onnx_model():
    net = nn.Linear(6, 1)
    X = torch.randn(6)
    
    onnx_path = os.path.join(TEST_FILES_DIR, 'model.onnx')
    torch.onnx.export(net, X, onnx_path)
    onnx_model = onnx.load_model(onnx_path)

    signature = infer_signature(X.numpy(), net(X).detach().numpy())
    model_info = mlflow.onnx.log_model(onnx_model, 
                                       "model_onnx", 
                                       signature=signature,
                                       registered_model_name='registered_model_onnx')



# container_id = start_triton_server('model_repository')
# print(container_id)

# stop_docker_container(container_id)

# print(kill_mlflow())

if __name__ == '__main__':
    pytest.main([__file__]) 
    stop_docker_container('triton_server')   
    

