from kfp import dsl
import pipeline_config
# from steps.dataloader_step import dataloader_step


@dsl.component(
    base_image = pipeline_config.BaseImages.DATALOADER_IMAGE,
    packages_to_install = pipeline_config.Dependencies.DATALOADER_PACKAGES,
    # base_image="python:3.9",
    # packages_to_install=[
    #     "pandas",
    #     "numpy",
    #     "pytorch_forecasting",
    #     "torch",
    #     "google-cloud-storage"
    # ],
    target_image = pipeline_config.TargetImages.DATALOADER_IMAGE
)
def dataloader_component(
    project: str,
    preprocessed_data_input: dsl.Input[dsl.Dataset],
    special_days: list,
    batch_size: int,
    training_output: dsl.Output[dsl.Dataset],
    train_loader_output: dsl.Output[dsl.Dataset],
    val_loader_output: dsl.Output[dsl.Dataset],
    gcs_bucket: str,
    max_prediction_length: int,
    max_encoder_length: int
):
    
    from steps.dataloader_step import dataloader_step
    
    dataloader_step(
        project = project,
        preprocessed_data_input = preprocessed_data_input.path,
        special_days = special_days,
        batch_size = batch_size,
        training_output = training_output.path,
        train_loader_output = train_loader_output.path,
        val_loader_output = val_loader_output.path,
        gcs_bucket = gcs_bucket,
        max_prediction_length = max_prediction_length,
        max_encoder_length = max_encoder_length
    )
   