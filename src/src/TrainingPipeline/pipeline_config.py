class Root:
    PIPELINE_NAME = "lightweight-python-components-pipeline"
    PIPELINE_ROOT = f"gs://vertex-pipeline-roots"

class ProjectConfig:
    PROJECT_ID = "ritheesh-1733201347"
    LOCATION = "us-central1"
    BUCKET_NAME = "stallion-data"
    DATA_PATH = "merged_dataset.csv"
    
class Dependencies:
    PREPROCESS_PACKAGES = ["pandas", "google-cloud-storage", "numpy"]
    DATALOADER_PACKAGES = ["pandas", "numpy", "pytorch_forecasting", "torch", "google-cloud-storage"]
    HPT_PACKAGES = ["pytorch_forecasting", "torch", "lightning", "optuna", "statsmodels", "optuna-integration[pytorch_lightning]", "tensorboard", "google-cloud-storage"]
    TRAINING_PACKAGES = ["pytorch_forecasting", "torch", "lightning", "tensorboard", "pytorch_optimizer", "google-cloud-storage"]

class BaseImages:
    PREPROCESS_IMAGE = "python:3.9"
    DATALOADER_IMAGE = "python:3.9"
    HPT_IMAGE = "python:3.9"
    TRAINING_IMAGE = "python:3.9"
    
class TargetImages:
    PREPROCESS_IMAGE = 'us-central1-docker.pkg.dev/ritheesh-1733201347/ngd/preprocess_component'
    DATALOADER_IMAGE = "us-central1-docker.pkg.dev/ritheesh-1733201347/ngd/dataloader_component"
    HPT_IMAGE = 'us-central1-docker.pkg.dev/ritheesh-1733201347/ngd/hpt_component'
    TRAINING_IMAGE = 'us-central1-docker.pkg.dev/ritheesh-1733201347/ngd/training_component'
    
class ComputeResources:
    PREPROCESS_MACHINE_TYPE = "n1-standard-32"
    DATALOADER_MACHINE_TYPE = ""
    HPT_MACHINE_TYPE = ""
    TRAINING_MACHINE_TYPE = ""