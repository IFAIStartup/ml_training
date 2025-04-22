import os
import mlflow
import onnx
import onnxsim
from mlflow.models import infer_signature
import torch
from pytorch_lightning.callbacks import Callback
from typing import Tuple
from ml_training.wrappers.deeplab.model import DeepLabV3PlusModule
from ml_training.utils import log_and_register_mlflow_model


class PostTrainCallback(Callback):
    """Callback that exports best checkpoint to ONNX 
    and logs model to mlflow server (logging formats - pytorch, onnx)
    """
    
    def __init__(self, export_path, config, registered_model_name):
        self.export_path = export_path
        self.registered_model_name = registered_model_name
        self.classes = config['DATA']['classes']
        self.channels = config['DATA']['channels']
        self.img_size = config['DATA']['img_size']
        self.encoder = config['TRAIN']['encoder']

    def on_train_end(self, trainer, pl_module):
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        onnx_path = os.path.join(self.export_path, 'model.onnx')

        model = self._load_best_model(best_model_path)
        dummy_in, dummy_out = self._get_dummy_in_out(model)
        self._export_model_to_onnx(model, dummy_in, dummy_out, onnx_path)

        dummy_in = {'input': dummy_in.cpu().numpy()}
        dummy_out = {'output': dummy_out.cpu().numpy()}
        log_and_register_mlflow_model(onnx_path, dummy_in, dummy_out, self.registered_model_name)     

        
    def _load_best_model(self, best_model_path) -> DeepLabV3PlusModule:
        
        model = DeepLabV3PlusModule.load_from_checkpoint(
            encoder=self.encoder,
            checkpoint_path=best_model_path,
            classes=self.classes,
            strict=False
        )

        model = model.to('cuda')
        model.eval()

        return model

    def _get_dummy_in_out(self, model: DeepLabV3PlusModule) -> Tuple[torch.Tensor, torch.Tensor]:
        dummy_in = torch.randn(1, self.channels, self.img_size, self.img_size, device='cuda')
        with torch.no_grad():
            dummy_out = model(dummy_in)
        
        return dummy_in, dummy_out
    
    def _export_model_to_onnx(self, model, dummy_in, dummy_out, onnx_path):
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(model.to('cpu'), dummy_in.to('cpu'), onnx_path, export_params=True, verbose=False, input_names=input_names,
                          output_names=output_names, opset_version=17)

        onnx_model = onnx.load(onnx_path)
        onnx_model, _ = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, onnx_path)

    def log_model_to_mlflow(self, model, onnx_path, signature):
        onnx_model = onnx.load(onnx_path)
        model_info = mlflow.pytorch.log_model(model, "model_pytorch", signature=signature)
        mlflow.onnx.log_model(onnx_model, "model_onnx", signature=signature,
                              registered_model_name=self.registered_model_name)
