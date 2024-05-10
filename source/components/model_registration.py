import mlflow
from source.entity import ModelRegistrationConfig
from tensorflow.keras.models import load_model

class ModelRegistation:

    def __init__(self,config: ModelRegistrationConfig, mlflow_run_id):
        self.config = config
        self.mlflow_run_id = mlflow_run_id 


    def register_model(self):

        with mlflow.start_run(run_id=self.mlflow_run_id,log_system_metrics=True):
            model_info = mlflow.tensorflow.log_model(
                model=self.config.model,
                artifact_path=self.config.model_registry_folder,
                signature=self.config.signature,
                registered_model_name=self.config.registered_model_name,
            )
        mlflow.end_run()

        return model_info
