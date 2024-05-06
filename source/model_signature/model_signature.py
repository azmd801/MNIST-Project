import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

import numpy as np
import mlflow

def generate_signature(trained_model_path, input_example):
    """
    Generates a model signature based on a provided model and an input example.

    Parameters:
    - model (tf.keras.Model): The trained model for which the signature needs to be generated.
    - input_example (numpy.ndarray): An example input data in the form of a NumPy array.

    Returns:
    - dict: A dictionary representing the inferred model signature.
    """
    model = load_model(trained_model_path)
    model_output = model.predict(input_example)
    signature = mlflow.models.infer_signature(model_input=input_example, model_output=model_output)
    
    return model,signature


