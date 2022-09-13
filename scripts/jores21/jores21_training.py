import os
import logging
import torch
import numpy as np
import pandas as pd
import eugene as eu

# Set-up directories
eu.settings.dataset_dir = "/cellar/users/aklie/data/eugene/jores21"
eu.settings.output_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/output/jores21"
eu.settings.logging_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/logs/jores21"
eu.settings.config_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/configs/jores21"
eu.settings.verbosity = logging.ERROR

# Load in the preprocessed training data
sdata_leaf = eu.dl.read(os.path.join(eu.settings.dataset_dir, "leaf_processed_train.h5sd"))
sdata_proto = eu.dl.read(os.path.join(eu.settings.dataset_dir, "proto_processed_train.h5sd"))
sdata_combined = eu.dl.concat([sdata_leaf, sdata_proto], keys=["leaf", "proto"])
sdata_leaf, sdata_proto, sdata_combined

# Grab initialization motifs
core_promoter_elements = eu.dl.motif.MinimalMEME(os.path.join(eu.settings.dataset_dir, 'CPEs.meme'))
tf_groups = eu.dl.motif.MinimalMEME(os.path.join(eu.settings.dataset_dir, 'TF-clusters.meme'))
all_motifs = {**core_promoter_elements.motifs, **tf_groups.motifs}

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
    
    seed_everything(seed)
    
    # Initialize the model prior to conv filter initialization
    eu.models.init_weights(model)

    # Initialize the conv filters
    if arch == "Jores21CNN":
        module_name, module_number, kernel_name, kernel_number = "biconv", None, "kernels", 0, 
    elif arch in ["CNN", "Hybrid"]:
        module_name, module_number, kernel_name, kernel_number = "convnet", 0, None, None
    eu.models.init_from_motifs(
        model, 
        all_motifs, 
        module_name=module_name,
        module_number=module_number,
        kernel_name=kernel_name,
        kernel_number=kernel_number
    )

    # Return the model
    return model 

# Train 5 models with 5 different random initializations
model_types = ["CNN", "Hybrid", "Jores21CNN"]
model_names = ["ssCNN", "ssHybrid", "Jores21CNN"]
trials = 5
for model_name, model_type in zip(model_names, model_types):
    for trial in range(1, trials+1):
        print(f"{model_name} trial {trial}")

        # Initialize the model
        leaf_model = prep_new_model(
            arch=model_type, 
            config=os.path.join(eu.settings.config_dir, f"{model_name}.yaml"),
            seed=trial
        )
        
        # Train the model
        eu.train.fit(
            model=leaf_model, 
            sdata=sdata_leaf, 
            gpus=1, 
            target_keys="enrichment",
            train_key="train_val",
            epochs=25,
            batch_size=128,
            num_workers=0,
            name=model_name,
            seed=trial,
            version=f"leaf_trial_{trial}",
            enable_model_summary=False,
            verbosity=logging.ERROR
        )
        
        # Get predictions on the training data
        eu.evaluate.train_val_predictions(
            leaf_model,
            sdata=sdata_leaf, 
            target_keys="enrichment",
            train_key="train_val",
            batch_size=128,
            num_workers=0,
            name=model_name,
            version=f"leaf_trial_{trial}",
            prefix=f"{model_name}_trial_{trial}_"
        )
        
        # Delete to make room for new model
        del leaf_model
        
# Save training predictions        
sdata_leaf.write_h5sd(os.path.join(eu.settings.output_dir, "leaf_train_predictions.h5sd"))

# Train 5 models with 5 different random initializations
model_types = ["CNN", "Hybrid", "Jores21CNN"]
model_names = ["ssCNN", "ssHybrid", "Jores21CNN"]
trials = 5
for model_name, model_type in zip(model_names, model_types):
    for trial in range(1, trials+1):
        print(f"{model_name} trial {trial}")

        # Initialize the model
        proto_model = prep_new_model(
            arch=model_type, 
            config=os.path.join(eu.settings.config_dir, f"{model_name}.yaml"),
            seed=trial
        )
        # Train the model
        eu.train.fit(
            model=proto_model, 
            sdata=sdata_proto, 
            gpus=1, 
            target_keys="enrichment",
            train_key="train_val",
            epochs=25,
            batch_size=128,
            num_workers=0,
            name=model_name,
            seed=trial,
            version=f"proto_trial_{trial}",
            enable_model_summary=False,
            verbosity=logging.ERROR
        )
        
        # Get predictions on the training data
        eu.evaluate.train_val_predictions(
            proto_model,
            sdata=sdata_proto, 
            target="enrichment",
            train_key="train_val",
            batch_size=128,
            num_workers=0,
            name=model_name,
            version=f"proto_trial_{trial}",
            prefix=f"{model_name}_trial_{trial}_"
        )
        
        # Delete to make room for new model
        del proto_model
        
# Save training predictions
sdata_proto.write_h5sd(os.path.join(eu.settings.output_dir, "proto_train_predictions.h5sd"))

# Train 5 models with 5 different random initializations
model_types = ["CNN", "Hybrid", "Jores21CNN"]
model_names = ["ssCNN", "ssHybrid", "Jores21CNN"]
trials = 5
for model_name, model_type in zip(model_names, model_types):
    for trial in range(1, trials+1):
        print(f"{model_name} trial {trial}")

        # Initialize the model
        combined_model = prep_new_model(
            arch=model_type, 
            config=os.path.join(eu.settings.config_dir, f"{model_name}.yaml"),
            seed=trial
        )
        
        # Train the model
        eu.train.fit(
            model=combined_model, 
            sdata=sdata_combined, 
            gpus=1, 
            target_keys="enrichment",
            train_key="train_val",
            epochs=25,
            batch_size=128,
            num_workers=0,
            name=model_name,
            seed=trial,
            version=f"combined_trial_{trial}",
            enable_model_summary=False,
            verbosity=logging.ERROR
        )
        
        # Get predictions on the training data
        eu.evaluate.train_val_predictions(
            combined_model,
            sdata=sdata_combined, 
            target="enrichment",
            train_key="train_val",
            batch_size=128,
            num_workers=0,
            name=model_name,
            version=f"combined_trial_{trial}",
            prefix=f"{model_name}_trial_{trial}_"
        )
        
        # Delete to make room for new model
        del combined_model
        
# Save training predictions
sdata_combined.write_h5sd(os.path.join(eu.settings.output_dir, "combined_train_predictions.h5sd"))