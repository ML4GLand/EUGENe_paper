# General imports
import os
import glob
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning
from tqdm.auto import tqdm

# EUGENe imports and settings
import eugene as eu
from eugene import models
from eugene.models import zoo
from eugene import evaluate
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/ray13"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/revision/ray13"
settings.logging_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/logs/revision/ray13"
settings.figure_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/figures/revision/ray13"

# EUGENe packages
import seqdata as sd

# ray13 helpers
sys.path.append("/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/scripts/ray13")
from ray13_helpers import rnacomplete_metrics_sdata_table

# For changable illustrator text
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Print versions
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Eugene version: {eu.__version__}")
#print(f"SeqData version: {sd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")


###########
# Load data
###########

# Set the number of kmers to use
number_kmers=None

# Load the test data
sdata_test = sd.open_zarr(os.path.join(settings.dataset_dir, "norm_setB_ST.zarr"))
keys = pd.Index(sdata_test.data_vars.keys())
target_mask = keys.str.contains("RNCMPT")
target_cols = keys[target_mask]

# Load in the Set B presence/absence predictions
b_presence_absence = np.load(os.path.join(settings.dataset_dir, "setB_binary.npy"))
setB_observed = sdata_test[target_cols]

# Also need the multi-task columns (single task we could train on all the columns)
sdata_training = sd.open_zarr(os.path.join(eu.settings.dataset_dir, "norm_setA_sub_MT.zarr"))
keys_MT = pd.Index(sdata_training.data_vars.keys())
target_mask_MT = keys_MT.str.contains("RNCMPT")
target_cols_MT = keys_MT[target_mask_MT]
del sdata_training

# Which model version
version = 0

#######################
# Multitask performance
#######################

"""
# Get predictions on the test data from all multi task models

print(f"Testing DeepBind MultiTask version {version} model")
arch = models.zoo.DeepBind(
    input_len=41, # Length of padded sequences
    output_dim=len(target_cols_MT), # Number of multitask outputs
    conv_kwargs=dict(input_channels=4, conv_channels=[1024], conv_kernels=[16], dropout_rates=0.25, batchnorm=0.25),
    dense_kwargs=dict(hidden_dims=[512], dropout_rates=0.25, batchnorm=True),
)
model_file = glob.glob(os.path.join(eu.settings.logging_dir, "DeepBind_MT", f"v{version}", "checkpoints", "*"))[0]
model = models.SequenceModule.load_from_checkpoint(model_file, arch=arch)
evaluate.predictions_sequence_module(
    model,
    sdata=sdata_test,
    seq_key="ohe_seq",
    target_keys=target_cols_MT,
    batch_size=1024,
    num_workers=4,
    prefetch_factor=2,
    in_memory=True,
    transforms={"ohe_seq": lambda x: torch.tensor(x, dtype=torch.float32), "target": lambda x: torch.tensor(x, dtype=torch.float32)},
    name="DeepBind_MT",
    version=f"v{version}",
    file_label="test",
    suffix="_MT"
)
del model

sd.to_zarr(sdata_test, os.path.join(settings.output_dir, f"norm_test_predictions_v{version}_MT.zarr"), load_first=True, mode="w")
"""

################
# Saving results
################

# Save the sdata with the predictions for single task, and multitask models
sdata_test = sd.open_zarr(os.path.join(settings.output_dir, f"norm_test_predictions_v{version}_MT.zarr"))

# Get evaluation metrics for all single task models and format for plotting
pearson_MT_df, spearman_MT_df = rnacomplete_metrics_sdata_table(sdata_test, b_presence_absence, target_cols_MT, verbose=False, swifter=True, num_kmers=number_kmers, preds_suffix="_predictions_MT")
pearson_MT_long = pearson_MT_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_MT_long = spearman_MT_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_MT_long["Model"] = "MultiTask"
spearman_MT_long["Model"] = "MultiTask"
pearson_MT_long.to_csv(os.path.join(settings.output_dir, f"pearson_performance_{number_kmers}kmers_MT.tsv"), index=False, sep="\t")
spearman_MT_long.to_csv(os.path.join(settings.output_dir, f"spearman_performance_{number_kmers}kmers_MT.tsv"), index=False, sep="\t")

# Plot just the multi task model eval
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_MT_long, x="Metric", y="Pearson", color="blue", ax=ax[0])
sns.boxplot(data=spearman_MT_long, x="Metric", y="Spearman", color="blue", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(settings.figure_dir, f"correlation_boxplots_{number_kmers}kmers_MT.pdf"))
