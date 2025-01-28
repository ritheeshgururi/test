import pandas as pd
import numpy as np
from google.cloud import storage
import logging
from utils.data_utils import (download_file_from_gcs, upload_file_to_gcs)


def preprocess_step(
    project,
    gcs_bucket,
    gcs_path,
    special_days,
    preprocessed_data_path
):
    """Fetches and preprocesses the Volume Forecasting Data"""
    
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting preprocessing")
    
#     #download and process data
#     client = storage.Client(project=project)
#     bucket = client.bucket(gcs_bucket)
#     blob = bucket.blob(gcs_path)
    
    
#     local_path = "/tmp/raw_data.csv"
#     blob.download_to_filename(local_path)

    local_path = "/tmp/raw_data.csv"
    download_file_from_gcs(project, gcs_bucket, gcs_path, local_path)
    
    data = pd.read_csv(local_path)
    
    data['date'] = pd.to_datetime(data['date'])
    
    #add time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    
    #add features
    data["month"] = data.date.dt.month.astype(str).astype("category")
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
    data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")
    
    data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    
    #saving to component output
    data.to_pickle(preprocessed_data_path.path)
    
    # #saving to GCS
    # preprocessed_gcs_path = f"artifacts/preprocessed_data.pkl"
    # blob = bucket.blob(preprocessed_gcs_path)
    # blob.upload_from_filename(preprocessed_data_path.path)
    
    preprocessed_gcs_path = f"artifacts/preprocessed_data.pkl"
    upload_file_to_gcs(project, gcs_bucket, preprocessed_data_path.path, preprocessed_gcs_path)
    
    logger.info(f"Preprocessing completed and saved to GCS: gs://{gcs_bucket}/{preprocessed_gcs_path}")