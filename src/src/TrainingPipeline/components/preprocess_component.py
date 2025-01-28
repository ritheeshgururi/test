from kfp import dsl
import pipeline_config
# from steps.preprocess_step import preprocess_step


@dsl.component(
    base_image = pipeline_config.BaseImages.PREPROCESS_IMAGE,
    packages_to_install = pipeline_config.Dependencies.PREPROCESS_PACKAGES,
    target_image = pipeline_config.TargetImages.PREPROCESS_IMAGE
)
def preprocess_component(
    project: str,
    location: str,
    gcs_bucket: str,
    gcs_path: str,
    special_days: list,
    preprocessed_data: dsl.Output[dsl.Dataset]
):
    
    # from steps.preprocess_step import preprocess_step
    
    preprocess_step(
        project =project,
        gcs_bucket = gcs_bucket,
        gcs_path = gcs_path,
        special_days = special_days,
        preprocessed_data_path = preprocessed_data.path
    )