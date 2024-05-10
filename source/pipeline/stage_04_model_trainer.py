from source.config.configuration import ConfigurationManager
from source.components.model_trainer import ModelTrainer
from source.logging import logger
from tensorflow.keras.models import Sequential


class ModelTrainerTrainingPipeline:
    def __init__(self,training_data):
        self.training_data = training_data

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(
            config=model_trainer_config,
            train_data=self.training_data
            )
        # print(model_trainer)
        return model_trainer.train()