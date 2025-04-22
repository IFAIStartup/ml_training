import os
import glob
import yaml
import shutil
import json
import requests
import mlflow
from typing import List
from mlflow import MlflowClient
from ml_training.utils import dict_to_pbtxt, DataType


class TritonDeploymentClient:
    def __init__(self, triton_model_repository: str, triton_url: str):
        self.triton_model_repository = triton_model_repository
        self.triton_url = triton_url
    
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        Deploy a model to the specified target.
        In the case of conflicts (e.g. if it's not possible to create the specified deployment
        without due to conflict with an existing deployment), raises a
        :py:class:`mlflow.exceptions.MlflowException`. 

        Args:
            name: Unique name to use for deployment. If another deployment exists with the same
                name, raises a :py:class:`mlflow.exceptions.MlflowException`
            model_uri: URI of model to deploy
            flavor: (optional) Model flavor to deploy. If unspecified, a default flavor
                will be chosen.
            config: (optional) Dict containing updated target-specific configuration for the
                deployment
            endpoint: (optional) Endpoint to create the deployment under. May not be supported
                by all targets

        Returns:
            Dict corresponding to created deployment, which must contain the 'name' key.

        """
        # model_source = self._search_registered_model(model_uri)
        # if model_source is None:
        #     raise ValueError

        # model_path = self._get_model_path(model_source)
        model_path = self._get_model_path(model_uri)
        if model_path is None:
            raise ValueError

        self._triton_unload_model(name)
        self._create_triton_model(model_path, name)
        self._triton_load_model(name)

    def delete_deployment(self, name, config=None, endpoint=None):
        """Delete the deployment with name ``name`` from the specified target.

        Args:
            name: Name of deployment to delete
            config: (optional) dict containing updated target-specific configuration for the
                deployment
            endpoint: (optional) Endpoint containing the deployment to delete. May not be
                supported by all targets

        Returns:
            None
        """
        self._triton_unload_model(name)
        model_path = os.path.join(self.triton_model_repository, name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

    def list_deployments(self, endpoint=None) -> List[dict]:
        """List deployments.

        This method returns an unpaginated list of all deployments.

        Args:
            endpoint: (optional) List deployments in the specified endpoint. May not be
                supported by all targets

        Returns:
            A list of dicts corresponding to deployments. Each dict 
            contains a 'name' key containing the deployment name.
        """
        return self._triton_model_index()

    def get_deployment(self, name, endpoint=None):
        """
        Returns a dictionary describing the specified deployment, throwing a
        :py:class:`mlflow.exceptions.MlflowException` if no deployment exists with the provided
        ID.
        The dict is guaranteed to contain an 'name' key containing the deployment name.
        The other fields of the returned dictionary and their types may vary across
        deployment targets.

        Args:
            name: ID of deployment to fetch.
            endpoint: (optional) Endpoint containing the deployment to get. May not be
                supported by all targets.

        Returns:
            A dict corresponding to the retrieved deployment. The dict is guaranteed to
            contain a 'name' key corresponding to the deployment name. The other fields of
            the returned dictionary and their types may vary across targets.
        """
        pass

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """Compute predictions on inputs using the specified deployment or model endpoint.

        Note that the input/output types of this method match those of `mlflow pyfunc predict`.

        Args:
            deployment_name: Name of deployment to predict against.
            inputs: Input data (or arguments) to pass to the deployment or model endpoint for
                inference.
            endpoint: Endpoint to predict against. May not be supported by all targets.

        Returns:
            A :py:class:`mlflow.deployments.PredictionsResponse` instance representing the
            predictions and associated Model Server response metadata.

        """
        pass


    def _search_registered_model(self, model_uri: str) -> str | None:

        model = mlflow.pyfunc.load_model(model_uri)
        metadata = model.metadata
        run_id = metadata.run_id
        client = MlflowClient()
        
        # Get all versions of the model filtered by name
        filter_string = f"run_id='{run_id}'"
        results = client.search_model_versions(filter_string)
        results.sort(key=lambda x: -int(x.version))
        
        model_source = None
        for res in results:
            model_source = res.source
            break
        
        return model_source
    
    def _create_triton_model(self, model_path: str, name: str, pbtxt: bool = False):
        triton_model_path = os.path.join(self.triton_model_repository, name)
        self._copy_model_to_triton(model_path, triton_model_path)
        if pbtxt:
            self._create_pbtxt_cfg(model_path, triton_model_path)
        
    def _copy_model_to_triton(self, model_path: str, triton_model_path: str):
        triton_model_path = os.path.join(triton_model_path, '1')
        shutil.copytree(model_path, triton_model_path)
    
    def _create_pbtxt_cfg(self, model_path: str, triton_model_path: str):
        mlmodel_path = os.path.join(model_path, 'MLmodel')
        with open(mlmodel_path) as f:
            meta = yaml.safe_load(f)
        
        signature = meta['signature']

        inputs = []
        for sign_input in json.loads(signature['inputs']):
            dims = sign_input['tensor-spec']['shape']
            dims[0] = 1
            inp = {
                'name': sign_input['name'],
                'data_type': DataType.TYPE_FP32,    # TODO: replace with datatype from signature
                'dims': dims,
            }
            inputs.append(inp)

        outputs = []
        for sign_output in json.loads(signature['outputs']):
            dims = sign_output['tensor-spec']['shape']
            dims[0] = 1
            out = {
                'name': sign_output['name'],
                'data_type': DataType.TYPE_FP32, # TODO
                'dims': dims,
            }
            outputs.append(out)
        
        pbtxt_dict = {
            "name": os.path.basename(triton_model_path),
            "backend": "onnxruntime",
            "max_batch_size": 0,
            "input": inputs,
            "output": outputs,
        }

        with open(os.path.join(triton_model_path, 'config.pbtxt'), 'w') as f:
            txt = dict_to_pbtxt(pbtxt_dict)
            f.write(txt)
    

    def _get_model_path(self, model_uri: str) -> str | None:
        # model_info = mlflow.models.get_model_info(model_uri)
        # artifacts_dir = './mlartifacts'
        # model_paths = glob.glob(os.path.join(artifacts_dir, '*', model_info.run_id, model_info.artifact_path))
        # if len(model_paths) == 0:
        #     return None
        
        # model_path = model_paths[0]
        # return model_path

        model_source = self._search_registered_model(model_uri)
        if model_source is None:
            return None
        
        artifacts_dir = './mlartifacts'
        relative_model_path = model_source.replace('mlflow-artifacts:/', '')
        # relative_model_path = relative_model_path.replace('file://', '')
        model_path = os.path.join(artifacts_dir, relative_model_path)

        return model_path
    
    def _triton_model_index(self) -> dict | list:
        response = requests.post(self.triton_url + '/v2/repository/index')
        triton_models = json.loads(response.text)
        return triton_models

    def _triton_load_model(self, model_name: str):
        requests.post(self.triton_url + f'/v2/repository/models/{model_name}/load')

    def _triton_unload_model(self, model_name: str):
        requests.post(self.triton_url + f'/v2/repository/models/{model_name}/unload')


# [{"name":"buildings360_r50_250923","version":"1","state":"READY"},{"name":"effnetb0_051023","version":"1","state":"READY"},{"name":"yolov8x_seg_360_1280_dataset_080123","version":"1","state":"READY"}]
# client = TritonDeploymentClient('http://0.0.0.0:5000', './model_repository', 'http://0.0.0.0:8000')
# client.create_deployment('model', 'models:/default_model@last')


