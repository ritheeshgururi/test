from kfp import dsl
import pipeline_config
# from steps.hpt_step import hpt_step

@dsl.component(
    base_image = pipeline_config.BaseImages.HPT_IMAGE,
    packages_to_install = pipeline_config.Dependencies.HPT_PACKAGES,
    # base_image="python:3.9",
    # packages_to_install=[
    #     "pytorch_forecasting",
    #     "torch",
    #     "lightning",
    #     "optuna",
    #     "statsmodels", 
    #     "optuna-integration[pytorch_lightning]",
    #     "tensorboard",
    #     "google-cloud-storage"
    # ]
    target_image = pipeline_config.TargetImages.HPT_IMAGE
)
def hpt_component(
    train_loader_input: dsl.Input[dsl.Dataset],
    val_loader_input: dsl.Input[dsl.Dataset],
    best_params_output: dsl.Output[dsl.Artifact],
    project: str,
    gcs_bucket: str,
    n_trials: int,
    max_epochs: int
):
    from steps.hpt_step import hpt_step
    
    tune_hyperparameters(
        train_loader_input = train_loader_input.path,
        val_loader_input = val_loader_input.path,
        best_params_output = best_params_output.path,
        project = project,
        gcs_bucket = gcs_bucket,
        n_trials = n_trials,
        max_epochs = max_epochs
    )