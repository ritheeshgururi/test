from kfp import dsl
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

from components.preprocessing import preprocess as PreprocessOp
