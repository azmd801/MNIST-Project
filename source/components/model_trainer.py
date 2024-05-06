from source.utils.common import yaml_to_bytes,read_yaml
from source.entity import ModelTrainerConfig
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from pathlib import Path
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, train_data):
        self.config = config
        self.train_data = train_data

    def model_initialization(self):
        # architecture_config = self.config.architecture_config_dir
        compiler_config = self.config.compilation_config_dir
        # print(yaml_to_bytes(architecture_config))
        # model=keras.models.model_from_json(yaml_to_bytes(architecture_config))
        input_shape = (28, 28, 1)
        num_classes = 10
        model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
        model.compile(**read_yaml(compiler_config))

        return model


    def train(self):
        model = self.model_initialization()
        model.fit(x=self.train_data, epochs=3)

        ## Save model
        model.save(os.path.join(self.config.root_dir,self.config.saved_trained_model))

