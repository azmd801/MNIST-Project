from source.config.configuration import ConfigurationManager
from source.components.model_evaluation import ModelEvaluation
from source.logging import logger




class ModelEvaluationTrainingPipeline:
    def __init__(self,test_data):
        self.test_data = test_data
        

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(
            config=model_evaluation_config,
            test_data=self.test_data
            )
        model_evaluation.evaluate()

        return model_evaluation.is_model_acceptable()

