# General imports
import os
import sys
import torch
import numpy as np
import pandas as pd
import pytorch_lightning

# EUGENe imports and settings
import eugene as eu
from eugene import models
from eugene.models import zoo
from eugene import train
from eugene import evaluate
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/ray13"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/revision/ray13"
settings.logging_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/logs/revision/ray13"
settings.config_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/configs/ray13"

# EUGENe packages
import seqdata as sd

# Print versions
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Eugene version: {eu.__version__}")
#print(f"SeqData version: {sd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")

# Load in the training SetA processed data for single task and multitask models
sdata_training_ST = sd.open_zarr(os.path.join(settings.dataset_dir, "norm_setA_ST.zarr"))

# Grab the prediction columns for single task and multitask
ST_keys = pd.Index(sdata_training_ST.data_vars.keys())
target_mask_ST = ST_keys.str.contains("RNCMPT")
target_cols_ST = ST_keys[target_mask_ST]

# Instantiation function
from pytorch_lightning import seed_everything
def prep_new_model(
    seed,
    conv_dropout = 0,
    dense_dropout = 0,
    batchnorm = True
):
    # Set a seed
    seed_everything(seed)

    model = models.zoo.DeepBind(
        input_len=41, # Length of padded sequences
        output_dim=1, # Number of multitask outputs
        conv_kwargs=dict(input_channels=4, conv_channels=[16], conv_kernels=[16], dropout_rates=conv_dropout, batchnorm=batchnorm),
        dense_kwargs=dict(hidden_dims=[32], dropout_rates=dense_dropout, batchnorm=batchnorm),
    )
    
    # Initialize the model prior to conv filter initialization
    models.init_weights(model)

    module = models.SequenceModule(
        arch=model,
        task="regression",
        loss_fxn="mse",
        optimizer="adam",
        optimizer_lr=0.0005,
        scheduler_kwargs=dict(patience=2)
    )

    # Return the model
    return module

# Train a model on each target prediction!
for i, target_col in enumerate(target_cols_ST):
    print(f"Training DeepBind SingleTask model on {target_col}")

    # Initialize the model
    model = prep_new_model(seed=i, conv_dropout=0.5, dense_dropout=0.5, batchnorm=True)

    # Fit the model
    train.fit_sequence_module(
        model,
        sdata_training_ST,
        seq_key="ohe_seq",
        target_keys=target_col,
        in_memory=True,
        train_key="train_val",
        epochs=25,
        batch_size=100,
        num_workers=4,
        prefetch_factor=2,
        drop_last=False,
        early_stopping_patience=3,
        name="DeepBind_ST",
        version=target_col,
        transforms={"ohe_seq": lambda x: torch.tensor(x, dtype=torch.float32), "target": lambda x: torch.tensor(x, dtype=torch.float32)},
        seed=i
    )

    evaluate.train_val_predictions_sequence_module(
        model,
        sdata=sdata_training_ST,
        seq_key="ohe_seq",
        target_keys=target_col,
        in_memory=True,
        train_key="train_val",
        batch_size=1024,
        num_workers=4,
        prefetch_factor=2,
        name="DeepBind_ST",
        version=target_col,
        transforms={"ohe_seq": lambda x: torch.tensor(x, dtype=torch.float32), "target": lambda x: torch.tensor(x, dtype=torch.float32)},
        suffix="_ST"
    )
    
# Save the predictions!
sd.to_zarr(sdata_training_ST, os.path.join(settings.output_dir, f"norm_setA_predictions_ST.zarr"), load_first=True, mode="w")
