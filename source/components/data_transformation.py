import os
from source.logging import logger
import tensorflow as tf
import tensorflow_datasets as tfds
from source.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def preprocess_fn(self,data):
        image = tf.cast(data["image"], tf.float32) / self.config.max_pixel_intensity
        label = data["label"]
        return (image, label)
    

    def convert(self):
        data = tfds.load(
            name=self.config.tf_dataset,
            download=False,
            data_dir=self.config.data_path
            )
        
        prefetched_train_data = data['train'].map(self.preprocess_fn).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        prefetched_test_data = data['test'].map(self.preprocess_fn).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return prefetched_train_data,prefetched_test_data
         


