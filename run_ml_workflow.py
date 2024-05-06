from source.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from source.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from source.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from source.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from source.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from source.pipeline.stage_06_model_registration import ModelRegistationPipeline
import mlflow
from source.logging import logger
from tensorflow.keras.models import Sequential


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




# STAGE_NAME = "Data Validation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_validation = DataValidationTrainingPipeline()
#    data_validation.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   train_data_prefetched,test_data_prefetched = data_transformation.main()
   logger.info(f"pre prcess data state {train_data_prefetched} is loaded in memory")
   logger.info(f"pre prcess data state {test_data_prefetched} is loaded in memory")
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Trainer stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainerTrainingPipeline(training_data=train_data_prefetched)
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Evaluation stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationTrainingPipeline(test_data=test_data_prefetched)
   is_model_accepted = model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed with acceptance staus of model = {is_model_accepted} <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model registration stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_registration = ModelRegistationPipeline()
   model_info=model_registration .main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed with with registering {model_info}  <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e





