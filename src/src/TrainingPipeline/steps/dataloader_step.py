import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import pickle
from google.cloud import storage
import logging
from utils.data_utils import upload_file_to_gcs

def dataloader_step(
    project,
    preprocessed_data_input,
    special_days,
    batch_size,
    training_output,
    train_loader_output,
    val_loader_output,
    gcs_bucket,
    max_prediction_length,
    max_encoder_length
):
    """Creates training and validation datasets."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating dataloaders")
    
    data = pd.read_pickle(preprocessed_data_input.path)
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx= "time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["agency", "sku"],
        static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        time_varying_known_categoricals=["special_days", "month"],
        variable_groups={"special_days": special_days},
        time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "volume",
            "log_volume",
            "industry_volume",
            "soda_volume",
            "avg_max_temp",
            "avg_volume_by_agency",
            "avg_volume_by_sku",
        ],
        target_normalizer=GroupNormalizer(
            groups=["agency", "sku"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    
    #saving to component outputs
    with open(training_output.path, 'wb') as f:
        pickle.dump(training, f)
    with open(train_loader_output.path, 'wb') as f:
        pickle.dump(train_dataloader, f)
    with open(val_loader_output.path, 'wb') as f:
        pickle.dump(val_dataloader, f)
    
#     #saving to GCS
#     client = storage.Client()
#     bucket = client.bucket(gcs_bucket)
    
#     #saving training dataset
#     training_gcs_path = "artifacts/training_dataset.pkl"
#     blob = bucket.blob(training_gcs_path)
#     blob.upload_from_filename(training_output.path)
    
#     #saving train dataloader
#     train_loader_gcs_path = "artifacts/train_dataloader.pkl"
#     blob = bucket.blob(train_loader_gcs_path)
#     blob.upload_from_filename(train_loader_output.path)
    
#     #saving val dataloader
#     val_loader_gcs_path = "artifacts/val_dataloader.pkl"
#     blob = bucket.blob(val_loader_gcs_path)
#     blob.upload_from_filename(val_loader_output.path)

    training_gcs_path = "artifacts/training_dataset.pkl"
    train_loader_gcs_path = "artifacts/train_dataloader.pkl"
    val_loader_gcs_path = "artifacts/val_dataloader.pkl"
    
    upload_file_to_gcs(project, gcs_bucket, training_output.path, training_gcs_path)
    upload_file_to_gcs(project, gcs_bucket, train_loader_output.path, train_loader_gcs_path)
    upload_file_to_gcs(project, gcs_bucket, val_loader_output.path, val_loader_gcs_path)
    
    logger.info("Dataloaders created and saved to GCS")