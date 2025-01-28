import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from google.cloud import storage
import logging
from utils.data_utils import upload_file_to_gcs


def hpt_step(
    train_loader_input,
    val_loader_input,
    best_params_output,
    project,
    gcs_bucket,
    n_trials,
    max_epochs
):
    """Tunes Hyperparameters using optuna"""    
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hyperparameter tuning")
    
    with open(train_loader_input.path, 'rb') as f:
        train_dataloader = pickle.load(f)
    with open(val_loader_input.path, 'rb') as f:
        val_dataloader = pickle.load(f)
    
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path= "optuna_test",
        n_trials=n_trials,
        max_epochs=max_epochs,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    
    #saving to component output
    with open(best_params_output.path, 'wb') as f:
        pickle.dump(study.best_trial.params, f)
    
#     #saving to GCS
#     client = storage.Client()
#     bucket = client.bucket(gcs_bucket)
#     best_params_gcs_path = "artifacts/best_params.pkl"
#     blob = bucket.blob(best_params_gcs_path)
#     blob.upload_from_filename(best_params_output.path)

    best_params_gcs_path = "artifacts/best_params.pkl"
    
    upload_file_to_gcs(project, gcs_bucket, best_params_output.path, best_params_gcs_path)
    
    logger.info(f"Best parameters saved to GCS: gs://{gcs_bucket}/{best_params_gcs_path}")