from source.config.configuration import ConfigurationManager
from source.components.model_registration import ModelRegistation
from source.logging import logger


class ModelRegistationPipeline:
    def __init__(self,mlflow_run_id):
        self.mlflow_run_id = mlflow_run_id

    def main(self):
        config = ConfigurationManager()
        model_registration_config = config.get_model_registration_config()
        model_registration = ModelRegistation(config=model_registration_config, mlflow_run_id=self.mlflow_run_id)
        model_info = model_registration.register_model()

        return model_info