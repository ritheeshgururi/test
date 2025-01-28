from kfp import dsl
import pipeline_config
# from steps.training_step import training_step

@dsl.component(
    base_image = pipeline_config.BaseImages.TRAINING_IMAGE,
    packages_to_install = pipeline_config.Dependencies.TRAINING_PACKAGES,
    # base_image="python:3.9",
    # packages_to_install=[
    #     "pytorch_forecasting",
    #     "torch",
    #     "lightning",
    #     "tensorboard",
    #     "pytorch_optimizer",
    #     "google-cloud-storage"
    # ]
    target_image = pipeline_config.TargetImages.TRAINING_IMAGE

)
def train_model(
    training_input: dsl.Input[dsl.Dataset],
    train_loader_input: dsl.Input[dsl.Dataset],
    val_loader_input: dsl.Input[dsl.Dataset],
    best_params_input: dsl.Input[dsl.Artifact],
    model_output: dsl.Output[dsl.Model],
    project: str,
    bucket_name: str,
    max_epochs: int
):
    from steps.training_step import training_step
    
    training_step(
        training_input = training_input.path,
        train_loader_input = train_loader_input.path,
        val_loader_input = val_loader_input.path,
        best_params_input = best_params_input.path,
        model_output = model_output.path,
        project = project,
        bucket_name = bucket_name,
        max_epochs = max_epochs
    )