import os
import logging
import torch
import numpy as np
import pandas as pd
import eugene as eu
from pytorch_lightning import seed_everything

# Set-up logging and other dirs
eu.settings.dataset_dir = "/cellar/users/aklie/data/eugene/ray13"
eu.settings.output_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/output/ray13"
eu.settings.logging_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/logs/ray13"
eu.settings.config_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/configs/ray13"

# Load in the training SetA processed data for single task and multitask models
sdata_training_ST = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, eu.settings.dataset_dir, "norm_setA_processed_ST.h5sd"))

# Grab the prediction columns for single task and multitask
target_mask_ST = sdata_training_ST.seqs_annot.columns.str.contains("RNCMPT")
target_cols_ST = sdata_training_ST.seqs_annot.columns[target_mask_ST]

# Instantiation and init function
def prep_new_model(
    seed,
    conv_dropout = 0,
    fc_dropout = 0,
    batchnorm = True
):
    model = eu.models.DeepBind(
        input_len=41, # Length of padded sequences
        output_dim=1, # Number of multitask outputs
        strand="ss",
        task="regression",
        mode="rbp",
        conv_kwargs=dict(channels=[4, 32], conv_kernels=[16], dropout_rates=conv_dropout, batchnorm=batchnorm),
        fc_kwargs=dict(hidden_dims=[64], dropout_rate=fc_dropout, batchnorm=batchnorm),
        optimizer="adam",
        lr=0.0005,
        scheduler_patience=2
    )

    # Set a seed
    seed_everything(seed)
    
    # Initialize the model prior to conv filter initialization
    eu.models.init_weights(model)

    # Return the model
    return model 

# Train a model on each target prediction!
for i, target_col in enumerate(target_cols_ST):
    print(f"Training DeepBind SingleTask model on {target_col}")

    # Initialize the model
    model = prep_new_model(seed=i, conv_dropout=0.25, fc_dropout=0.25, batchnorm=True)

    try:
        # Train the model
        eu.train.fit(
            model=model, 
            sdata=sdata_training_ST, 
            gpus=1, 
            target_keys=target_col,
            train_key="train_val",
            epochs=25,
            early_stopping_metric="val_loss",
            early_stopping_patience=3,
            batch_size=100,
            num_workers=0,
            name="DeepBind_ST",
            seed=i,
            version=target_col,
            verbosity=logging.ERROR
        )

        # Get predictions on the training data
        eu.evaluate.train_val_predictions(
            model,
            sdata=sdata_training_ST, 
            target_keys=target_col,
            train_key="train_val",
            batch_size=1024,
            num_workers=0,
            name="DeepBind_ST",
            suffix="_ST",
            version=target_col
        )
    except:
        print(f"Training model on {target_col} failed")
        
    # Make room for the next model!
    del model 
    
# Save all the predictions in single sdata
sdata_training_ST.write_h5sd(os.path.join(eu.settings.output_dir, "DeepBind_ST", "norm_training_predictions_ST.h5sd"))