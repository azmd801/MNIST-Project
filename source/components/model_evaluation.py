
import pandas as pd
from source.entity import ModelEvaluationConfig
from tensorflow.keras.models import load_model



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, test_data):
        self.config = config
        self.test_data = test_data

    def evaluate(self):

        # Evaluate the model
        model = load_model(self.config.model_path)
        test_loss,test_accuracy = model.evaluate(self.test_data)

        # Create a new DataFrame
        df = pd.DataFrame({'Test loss': test_loss, 'Test accuracy': test_accuracy},index=[0])
        
        # Save DataFrame to CSV file
        df.to_csv(self.config.metric_file_name , index=False)
        
        return test_accuracy 

    def is_model_acceptable(self):

        return self.evaluate() > 0.8
    




