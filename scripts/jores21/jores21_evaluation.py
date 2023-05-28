# General imports
import os
import glob
import torch
import numpy as np
import xarray as xr

# EUGENe imports and settings
from eugene import models
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/jores21"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/revision/jores21"
settings.logging_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/logs/revision/jores21"
settings.config_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/configs/jores21"

# EUGENe packages
import seqdata as sd

# For illustrator editing
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sdata_leaf = sd.open_zarr(os.path.join(settings.dataset_dir, "jores21_leaf_test.zarr"))
sdata_proto = sd.open_zarr(os.path.join(settings.dataset_dir, "jores21_proto_test.zarr"))
def concat_seqdatas(seqdatas, keys):
    for i, s in enumerate(seqdatas):
        s["batch"] = keys[i]
    return xr.concat(seqdatas, dim="_sequence")
sdata_combined = concat_seqdatas([sdata_leaf, sdata_proto], ["leaf", "proto"])

# Train 5 models with 5 different random initializations for each dataset and mopdel architecture
test_sets = {"leaf": sdata_leaf, "proto": sdata_proto, "combined": sdata_combined}
configs = ["cnn.yaml", "hybrid.yaml", "jores21_cnn.yaml", "deepstarr.yaml"]
trials = 5
for test_set in test_sets:
    sdata = test_sets[test_set]
    # Make an output directory for this dataset if it doesn't exist
    if not os.path.exists(os.path.join(settings.output_dir, test_set)):
        os.mkdir(os.path.join(settings.output_dir, test_set))
    for config in configs:
        model_name = config.split(".")[0]
        for trial in range(1, trials+1):
        
            # Print the model name
            print(f"{test_set} {model_name} trial {trial}")

            # Grab the best model from that training run
            model_file = glob.glob(os.path.join(settings.logging_dir, model_name, f"{test_set}_trial_{trial}", "checkpoints", "*"))[0]
            model = models.load_config(config_path=config)
            best_model = models.SequenceModule.load_from_checkpoint(model_file, arch=model.arch)

            # Grab the predictions on the test set
            ohe_seqs = sdata["ohe_seq"].to_numpy().transpose(0, 2, 1)
            preds = best_model.predict(ohe_seqs, batch_size=128).detach().numpy().squeeze()
            sdata[f"{model_name}_{test_set}_trial_{trial}_preds"] = xr.DataArray(preds, dims=["_sequence"])

    pred_keys = [k for k in sdata.data_vars.keys() if "preds" in k]
    target_keys = ["enrichment"]
    sdata = sdata.chunk()
    sdata[["id", *target_keys, *pred_keys]].to_dataframe().to_csv(os.path.join(settings.output_dir, test_set, f"jores21_{test_set}_test_predictions.tsv"), sep="\t", index=False)
    sdata.to_zarr(os.path.join(settings.output_dir, test_set, f"jores21_{test_set}_test_predictions.zarr"), mode="w")
