import os
import logging
import torch
import numpy as np
import pandas as pd
import eugene as eu

# Configure EUGENe 
eu.settings.dataset_dir = "/cellar/users/aklie/data/eugene/kopp21"
eu.settings.output_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/output/kopp21"
eu.settings.logging_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/logs/kopp21"
eu.settings.config_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/configs/kopp21"
eu.settings.verbosity = logging.ERROR

# Load in the preprocessed training data
sdata = eu.dl.read_h5sd(filename=os.path.join(eu.settings.dataset_dir, "jund_train_processed.h5sd"))

# Function to prepare a new model for training
from pytorch_lightning import seed_everything
def prep_new_model(
    seed,
    arch,
    config
):
    # Instantiate the model
    model = eu.models.load_config(
        arch=arch,
        model_config=config
    )

    # Set a seed
    seed_everything(seed)
    
    # Initialize the model prior to conv filter initialization
    eu.models.init_weights(model)

    # Return the model
    return model 


# Train 5 models with 5 different random initializations
model_types = ["Hybrid"]
model_names = ["dsHybrid"]
trials = 5
for model_name, model_type in zip(model_names, model_types):
    for trial in range(1, trials+1):
        print(f"{model_name} trial {trial}")

        # Initialize the model
        model = prep_new_model(
            arch=model_type, 
            config=os.path.join(eu.settings.config_dir, f"{model_name}.yaml"),
            seed=trial
        )

        # Train the model
        eu.train.fit(
            model=model, 
            sdata=sdata, 
            gpus=1, 
            target_keys="target",
            train_key="train_val",
            epochs=25,
            early_stopping_metric="val_loss",
            early_stopping_patience=5,
            batch_size=64,
            num_workers=0,
            name=model_name,
            seed=trial,
            version=f"trial_{trial}",
            verbosity=logging.ERROR
        )
        
        # Get predictions on the training data
        eu.evaluate.train_val_predictions(
            model,
            sdata=sdata, 
            target_keys="target",
            train_key="train_val",
            batch_size=512,
            num_workers=0,
            name=model_name,
            version=f"trial_{trial}",
            prefix=f"{model_name}_trial_{trial}_"
        )
        
        # Make room for the next model
        del model
        
sdata.write_h5sd(os.path.join(eu.settings.output_dir, "jund_train_predictions_Hybrid.h5sd"))