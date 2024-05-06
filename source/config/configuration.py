from source.constants import *
from source.utils.common import read_yaml, create_directories,yaml_to_bytes
from source.entity import (DataIngestionConfig,
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelEvaluationConfig,
                                   ModelRegistrationConfig
                                )
from pathlib import Path
from source.model_signature.model_signature import  generate_signature


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        # model_config_filepath = MODEL_CONFIG_FILE_PATH 
        ):

        self.config = read_yaml(config_filepath)
        # self.params = read_yaml(model_config_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            tf_dataset=config.tf_dataset,
            local_data_folder=config.local_data_folder,
        )

        return data_ingestion_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        # create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            data_path = config.data_path,
            tf_dataset = config.tf_dataset,
            max_pixel_intensity =config.max_pixel_intensity,
            batch_size =config.batch_size,
            )

        return data_transformation_config
    


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            saved_trained_model = Path(config.saved_trained_model),
            architecture_config_dir=config.architecture_config_dir,
            compilation_config_dir=config.compilation_config_dir

        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name
           
        )

        return model_evaluation_config

    def get_model_registration_config(self) -> ModelRegistrationConfig:
        config = self.config.model_registration

        model,signature = generate_signature(
            trained_model_path=config.model_path,
            input_example=INPUT_INFERENCE_EXAMPLE
            )


        model_registration_config = ModelRegistrationConfig(
            model=model,
            signature=signature,
            model_registry_folder=config.model_registry_folder,
            registered_model_name=config.registered_model_name
            )

        return model_registration_config