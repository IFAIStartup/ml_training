import mlflow
from typing import Tuple


class BaseWrapper:
    def __init__(self, config: dict, registered_model_name: str = None, run_id: str = None):
        self.config = config
        self.registered_model_name = registered_model_name
        self.run_id = run_id

        if self.run_id is None:
            mlflow.end_run()
            self.project_name, self.run_name = self._get_project_and_run_name()
            mlflow.set_experiment(self.project_name)
            run = mlflow.start_run(run_name=self.run_name)  # TODO: add handling if run id is None, but run is actually active
            self.run_id = run.info.run_id
        else:
            mlflow.end_run()
            run = mlflow.start_run(self.run_id)
            experiment = mlflow.get_experiment(run.info.experiment_id)
            self.project_name = experiment.name
            self.run_name = run.info.run_name
            

    def train(self):
        pass

    def _get_project_and_run_name(self) -> Tuple[str]:
        return 'project', 'name'

    
    
