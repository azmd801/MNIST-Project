import os
# import urllib.request as request
from source.logging import logger
from source.utils.common import isEmpty
from pathlib import Path
from source.entity import DataIngestionConfig
# import tensorflow as tf
import tensorflow_datasets as tfds


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        if isEmpty(self.config.local_data_folder):

            _, ds_info = tfds.load(
                name=self.config.tf_dataset,
                split=["train", "test"],
                shuffle_files=True,
                data_dir=self.config.local_data_folder
                )
            logger.info(f"{self.config.tf_dataset} downloaded! with following info: \n{ds_info}")

        else:
            logger.info(f"Fllowing folders {os.listdir(self.config.local_data_folder)} File already exists" )  

        
    