from source.config.configuration import ConfigurationManager
from source.components.data_transformation import DataTransformation
from source.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_data_prefetched,test_data_prefetched = data_transformation.convert()

        return train_data_prefetched,test_data_prefetched