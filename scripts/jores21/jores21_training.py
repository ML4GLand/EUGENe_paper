# General imports
import os
import sys
import torch
import numpy as np
import pandas as pd
from copy import deepcopy 
import pytorch_lightning

# EUGENe imports and settings
import eugene as eu
from eugene import dataload as dl
from eugene import models
from eugene import train
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/jores21"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/jores21"
settings.logging_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/logs/jores21"
settings.config_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/configs/jores21"

# EUGENe packages
import seqdata as sd
import motifdata as md

# Print versions
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Eugene version: {eu.__version__}")
print(f"SeqData version: {sd.__version__}")
print(f"MotifData version: {md.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")

# Load in the preprocessed training data
sdata_leaf = sd.open_zarr(os.path.join(settings.dataset_dir, "jores21_leaf_train.zarr"))
sdata_proto = sd.open_zarr(os.path.join(settings.dataset_dir, "jores21_proto_train.zarr"))
sdata_combined = dl.concat_seqdatas([sdata_leaf, sdata_proto], ["leaf", "proto"])

# Load in PFMs to initialize the 1st layer of the model with
core_promoter_elements = md.read_meme(os.path.join(settings.dataset_dir, "CPEs.meme"))
tf_clusters = md.read_meme(os.path.join(settings.dataset_dir, "TF-clusters.meme"))
all_motifs = deepcopy(core_promoter_elements)
for motif in tf_clusters:
    all_motifs.add_motif(motif)

# Function for instantiating a new randomly initialized model
def prep_new_model(
    config,
    seed
):
    # Instantiate the model
    model = models.load_config(config_path=config, seed=seed)
    
    # Initialize the model prior to conv filter initialization
    models.init_weights(model, initializer="kaiming_normal")

    # Initialize the conv filters
    if model.arch_name == "Jores21CNN":
        layer_name = "arch.biconv.kernels"
        list_index = 0
    elif model.arch_name in ["CNN", "Hybrid", "DeepSTARR"]:
        layer_name = "arch.conv1d_tower.layers.0"
        list_index = None
    models.init_motif_weights(
        model=model,
        layer_name=layer_name,
        list_index=list_index,
        initializer="xavier_uniform",
        motifs=all_motifs,
        convert_to_pwm=False,
        divide_by_bg=True,
        motif_align="left",
        kernel_align="left"
    )

    # Return the model
    return model 

# Train 5 models with 5 different random initializations for each dataset and mopdel architecture
training_sets = {"leaf": sdata_leaf, "proto": sdata_proto, "combined": sdata_combined}
configs = ["cnn.yaml", "hybrid.yaml", "jores21_cnn.yaml", "deepstarr.yaml"]
trials = 5
for training_set in training_sets:
    for trial in range(1, trials+1):
        for config in configs:

            # Print the model name
            sdata = training_sets[training_set]
            model_name = config.split(".")[0]
            print(f"{training_set} {model_name} trial {trial}")

            # Initialize the model
            model = prep_new_model(config, seed=trial)

            # Fit the model
            train.fit_sequence_module(
                model,
                sdata,
                seq_var="ohe_seq",
                target_vars=["enrichment"],
                in_memory=True,
                train_var="train_val",
                epochs=25,
                batch_size=128,
                num_workers=4,
                prefetch_factor=2,
                drop_last=False,
                name=model_name,
                version=f"{training_set}_trial_{trial}",
                seq_transforms={"ohe_seq": lambda x: torch.tensor(x, dtype=torch.float32).transpose(1, 2), "target": lambda x: torch.tensor(x, dtype=torch.float32)},
                seed=trial
            )

            # Make room for the next model 
            del model
