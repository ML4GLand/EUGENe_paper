import os
import logging
import torch
import numpy as np
import pandas as pd
import eugene as eu
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
    eu.models.base.init_weights(model)

    # Return the model
    return model 


# Configure EUGENe 
eu.settings.dataset_dir = "/cellar/users/aklie/data/eugene/kopp21"
eu.settings.output_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/output/kopp21"
eu.settings.logging_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/logs/kopp21"
eu.settings.config_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/configs/kopp21"
eu.settings.verbosity = logging.ERROR


sdata = eu.dl.read_h5sd(filename=os.path.join(eu.settings.dataset_dir, "jund_train_processed.h5sd"))
# Train 5 models with 5 different random initializations
model_types = ["FCN", "CNN", "RNN", "Hybrid", "Kopp21CNN"]
model_names = ["dsFCN", "dsCNN", "dsRNN", "dsHybrid", "Kopp21CNN"]
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

        if model_type == "RNN":
            t_kwargs = transform_kwargs={"transpose": False}
        else:
            t_kwargs = transform_kwargs={"transpose": True}

        # Train the model
        eu.train.fit(
            model=model, 
            sdata=sdata, 
            gpus=1, 
            target="target",
            train_key="train_val",
            epochs=30,
            early_stopping_metric="val_loss",
            early_stopping_patience=5,
            transform_kwargs=t_kwargs,
            batch_size=64,
            num_workers=4,
            name=model_name,
            seed=trial,
            version=f"trial_{trial}",
            verbosity=logging.ERROR
        )
        # Get predictions on the training data
        eu.settings.dl_num_workers = 0
        eu.predict.train_val_predictions(
            model,
            sdata=sdata, 
            target="target",
            train_key="train_val",
            transform_kwargs=t_kwargs,
            name=model_name,
            version=f"trial_{trial}",
            prefix=f"{model_name}_trial_{trial}_"
        )
        del model 
sdata.write_h5sd(os.path.join(eu.settings.output_dir, "train_predictions.h5sd"))