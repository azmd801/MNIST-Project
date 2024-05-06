from dataclasses import dataclass
from pathlib import Path
from mlflow.models.signature import ModelSignature
from tensorflow.keras.models import Sequential


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    tf_dataset: str
    local_data_folder: Path
    



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list



@dataclass(frozen=True)
class DataTransformationConfig:
    data_path: Path
    tf_dataset: str
    max_pixel_intensity: int
    batch_size: int
    


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    saved_trained_model: Path
    architecture_config_dir: Path
    compilation_config_dir: Path



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    metric_file_name: Path

@dataclass(frozen=True)
class ModelRegistrationConfig:
    model: Sequential
    signature: ModelSignature
    model_registry_folder:str
    registered_model_name: str