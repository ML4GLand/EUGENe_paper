# General imports
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from copy import deepcopy 
import pytorch_lightning
from itertools import product

# EUGENe imports and settings
import eugene as eu
from eugene import dataload as dl
from eugene import models, train, evaluate
from eugene.dataload._augment import RandomRC
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/kopp21"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/revision/kopp21"
settings.logging_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/logs/revision/kopp21"
settings.config_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/configs/kopp21"

# EUGENe packages
import seqdata as sd

# Print versions
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Eugene version: {eu.__version__}")
print(f"SeqData version: {sd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train and evaluate a model using EUGENe.')
parser.add_argument('--configs', type=str, help='comma-separated list of config file names to use')
args = parser.parse_args()

# Load in the preprocessed training data
sdata = sd.open_zarr(os.path.join(settings.dataset_dir, 'kopp21_train.zarr'))

# Function to instantiate a new model
def prep_new_model(
    config,
    seed,
):
    # Instantiate the model
    model = eu.models.load_config(
        config_path=config,
        seed=seed
    )
    
    # Initialize the model prior to conv filter initialization
    eu.models.init_weights(model)

    # Return the model
    return model 

# Train 5 models with 5 different random initializations
configs = args.configs.split(',')
trials = 5
models_trained = ""
for config, trial in product(configs, range(1, trials+1)):
    model_name = config.split('.')[0]
    print(model_name)
    if model_name not in models_trained:
        models_trained += model_name + '_'

    # Initialize the model
    model = prep_new_model(os.path.join(settings.config_dir, config), seed=trial)
    
    # Set-up the transforms
    transforms = {"target": lambda x: torch.tensor(x, dtype=torch.float32)}
    if model_name != 'kopp21_cnn':
        random_rc = RandomRC()
        def ohe_seq_transform(x):
            x = torch.tensor(x, dtype=torch.float32).swapaxes(1, 2)
            return random_rc(x)
        transforms["ohe_seq"] = ohe_seq_transform
    else:
        transforms["ohe_seq"] = lambda x: torch.tensor(x, dtype=torch.float32).swapaxes(1, 2)
        
    # Fit the model
    eu.train.fit_sequence_module(
        model,
        sdata,
        gpus=1,
        seq_key="ohe_seq",
        target_keys=["target"],
        in_memory=True,
        train_key="train_val",
        epochs=25,
        early_stopping_metric='val_loss_epoch',
        early_stopping_patience=5,
        batch_size=64,
        num_workers=4,
        prefetch_factor=2,
        drop_last=False,
        name=model_name,
        version=f"trial_{trial}",
        transforms=transforms,
        seed=trial,
    )
    
    # Evaluate the model on train and validation sets
    evaluate.train_val_predictions_sequence_module(
        model,
        sdata,
        seq_key="ohe_seq",
        target_keys=["target"],
        in_memory=True,
        train_key="train_val",
        batch_size=1024,
        num_workers=4,
        prefetch_factor=2,
        name=model_name,
        version=f"trial_{trial}",
        transforms=transforms,
        prefix=f"{model_name}_trial_{trial}_"
    )

    # Make room for the next model
    del model
    
# Save the predictions!
sd.to_zarr(sdata, os.path.join(settings.output_dir, f"train_predictions_{models_trained}models.zarr"), load_first=True, mode="w")
