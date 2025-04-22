import os
import numpy as np
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import onnx
import onnxruntime as ort
from ultralytics.utils import LOGGER, colorstr
from ultralytics import YOLO
from dotenv import dotenv_values
from pathlib import Path
from ml_training.utils import log_and_register_mlflow_model

PREFIX = colorstr("MLflow: ")
SANITIZE = lambda x: {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}

env = {
    **dotenv_values(".env"),  # load general environment variables
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URI'])


class YOLOCallback:

    def __init__(self, config: dict, registered_model_name: str, run_id: str = None):
        self.config = config
        self.registered_model_name = registered_model_name
        self.run_id = run_id

    def on_pretrain_routine_end(self, trainer):
        """
        Log training parameters to MLflow at the end of the pretraining routine.

        This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
        experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
        from the trainer.

        Args:
            trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

        Global:
            mlflow: The imported mlflow module to use for logging.

        Environment Variables:
            MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
            MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
            MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        """

        uri = env['MLFLOW_TRACKING_URI'] # or str(RUNS_DIR / "mlflow")
        LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
        #mlflow.set_tracking_uri(uri)

        mlflow.autolog()
        # try:
        
        active_run = mlflow.active_run() or mlflow.start_run(self.run_id)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        mlflow.log_params(dict(trainer.args))
        # except Exception as e:
        #     LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")


    def on_train_epoch_end(self, trainer):
        """Log training metrics at the end of each train epoch to MLflow."""
        if mlflow:
            mlflow.log_metrics(
                metrics=SANITIZE(trainer.label_loss_items(trainer.tloss, prefix="train")), step=trainer.epoch
            )
            mlflow.log_metrics(metrics=SANITIZE(trainer.lr), step=trainer.epoch)


    def on_fit_epoch_end(self, trainer):
        """Log training metrics at the end of each fit epoch to MLflow."""
        if mlflow:
            mlflow.log_metrics(metrics=SANITIZE(trainer.metrics), step=trainer.epoch)


    def on_train_end(self, trainer):
        """Log model artifacts at the end of the training."""
        if mlflow:
            mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
            for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
                if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                    mlflow.log_artifact(str(f))

            # mlflow.end_run()
            LOGGER.info(
                f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n"
                f"{PREFIX}disable with 'yolo settings mlflow=False'"
            )
        
        onnx_path = self._onnx_export(trainer)
        self._log_models_to_mlflow(onnx_path)

    
    def _log_models_to_mlflow(self, onnx_path: str):
        dummy_in, outs = self._get_dummy_ins_outs(onnx_path)
        dummy_in = {'images': dummy_in}
        # dummy_outs = {'output0': out0, 'output1': out1}
        dummy_outs = {f'output{i}': out for i, out in enumerate(outs)}
        
        log_and_register_mlflow_model(onnx_path, dummy_in, dummy_outs, self.registered_model_name)

        
    def _onnx_export(self, trainer):
        pt_path = str(trainer.best)
        model = YOLO(pt_path)
        model.export(format='onnx', dynamic=False, simplify=True, opset=17, half=True)
        onnx_path = os.path.splitext(pt_path)[0] + '.onnx'
        return onnx_path
        
    def _get_dummy_ins_outs(self, onnx_path: str):
        """Inference exported ONNX model and return dummy input an output
        """
        ort_sess = ort.InferenceSession(onnx_path)
        dummy_in = np.random.randn(1, 3, self.config['imgsz'], self.config['imgsz']).astype('float32')
        outs = ort_sess.run(None, {'images': dummy_in})
        return dummy_in, outs
    
