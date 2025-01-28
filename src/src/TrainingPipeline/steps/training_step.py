import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import torch
from google.cloud import storage
from datetime import datetime
import logging
from utils.data_utils import upload_file_to_gcs


def training_step(
    training_input,
    train_loader_input,
    val_loader_input,
    best_params_input,
    model_output,
    project,
    bucket_name,
    max_epochs
):
    """Trains the TFT model and saves it to GCS"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting model training")
    
    with open(training_input.path, 'rb') as f:
        training = pickle.load(f)
    with open(train_loader_input.path, 'rb') as f:
        train_dataloader = pickle.load(f)
    with open(val_loader_input.path, 'rb') as f:
        val_dataloader = pickle.load(f)
    with open(best_params_input.path, 'rb') as f:
        best_params = pickle.load(f)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger("lightning_logs")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator= "cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback],
        logger=tb_logger,
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=best_params['learning_rate'],
        hidden_size=best_params['hidden_size'],
        attention_head_size=best_params['attention_head_size'],
        dropout=best_params['dropout'],
        hidden_continuous_size=best_params['hidden_continuous_size'],
        loss=QuantileLoss(),
        log_interval=10,
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    #saving model state and configuration
    model_dict = {
        'state_dict': tft.state_dict(),
        'hparams': tft.hparams,
        'training_config': training.get_parameters()
    }
    
    #saving locally first
    local_path = "/tmp/model.pth"
    torch.save(model_dict, local_path)
    
    # #uploading to GCS
    # client = storage.Client(project=project_id)
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob('models/tft_model.pth')
    # blob.upload_from_filename(local_path)
    
    model_gcs_path = "models/tft_model.pth"
    
    upload_file_to_gcs(project, gcs_bucket, local_path, model_gcs_path)
    
    #saving the GCS path to the model output
    model_info = {
        'gcs_path': f'gs://{bucket_name}/models/tft_model.pth',
        'model_type': 'TemporalFusionTransformer',
        'training_timestamp': datetime.utcnow().isoformat()
    }
    with open(model_output.path, 'wb') as f:
        pickle.dump(model_info, f)
    
    logger.info(f"Model trained and saved to GCS: gs://{bucket_name}/models/tft_model.pth")