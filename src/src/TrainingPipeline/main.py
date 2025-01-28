
PROJECT_ID = "ritheesh-1733201347"
BUCKET_NAME = "stallion-data"
PIPELINE_ROOT = f"gs://vertex-pipeline-roots"
DATA_PATH = "merged_dataset.csv"
LOCATION = "us-central1"
from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform
import logging
from datetime import datetime
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from components.preprocess_component import preprocess_component as PreprocessOp
import pipeline_config
preprocessOP=create_custom_training_job_from_component(PreprocessOp,
    display_name='PreProcessing Pipeline',
    machine_type=pipeline_config.ComputeResources.PREPROCESS_MACHINE_TYPE,
)
@dsl.pipeline(
    name="tft-training-pipeline",
    description="End to end pipeline for training the Temporal Fusion Transformer model"
)
def training_pipeline(
    project: str,
    location: str,
    gcs_bucket: str,
    gcs_path: str
):
    """Main pipeline for training and inference"""
    
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest"
    ]
    
    #pipeline components
    preprocess_task = preprocessOP(
        project=project,
        location=location,
        gcs_bucket=gcs_bucket,
        gcs_path=gcs_path,
        special_days=special_days
    )
#compiling the pipeline
compiler = Compiler()
compiler.compile(
    pipeline_func=training_pipeline,
    package_path="tft_training_pipeline_latestt.json"
)
#creating and running the training pipeline
training_job = aiplatform.PipelineJob(
    display_name="tft-training-pipeline",
    template_path="tft_training_pipeline_latestt.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project": PROJECT_ID,
        "location": LOCATION,
        "gcs_bucket": BUCKET_NAME,
        "gcs_path": DATA_PATH
    }
)
aiplatform.init(project=PROJECT_ID, location=LOCATION)








training_job.run()