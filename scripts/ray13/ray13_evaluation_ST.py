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
from eugene import models, evaluate, settings
from eugene.models import zoo
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
print(f"SeqData version: {sd.__version__}")
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

#########################
# Single task performance
#########################

# Get predictions for all single task models
arch = zoo.DeepBind(
    input_len=41,
    output_dim=1,
    conv_kwargs=dict(input_channels=4, conv_channels=[16], conv_kernels=[16], dropout_rates=0.5, batchnorm=True),
    dense_kwargs=dict(hidden_dims=[32], dropout_rates=0.5, batchnorm=True),
)
trained_model_cols = []
for i, target_col in enumerate(target_cols):
    print(f"Testing DeepBind SingleTask model on {target_col}")
    try:
        model_file = glob.glob(os.path.join(settings.logging_dir, "DeepBind_ST", target_col, "checkpoints", "*"))[0]
        model = models.SequenceModule.load_from_checkpoint(model_file, arch=arch)
        trained_model_cols.append(target_col)
    except:
        print(f"No model trained for {target_col}")
        continue
    evaluate.predictions_sequence_module(
        model,
        sdata=sdata_test, 
        seq_var="ohe_seq",
        target_vars=target_col,
        batch_size=5096,
        num_workers=4,
        prefetch_factor=2,
        in_memory=True,
        transforms={"ohe_seq": lambda x: torch.tensor(x, dtype=torch.float32), "target": lambda x: torch.tensor(x, dtype=torch.float32)},
        name="DeepBind_ST",
        version=target_col,
        file_label="test",
        suffix="_ST"
    )
    del model
print(f"Successful predictions: {len(trained_model_cols)}")
sd.to_zarr(sdata_test, os.path.join(settings.output_dir, "norm_test_predictions_ST.zarr"), mode="w")

################
# Saving results
################

# Load the predictions
#sdata_test = sd.open_zarr(os.path.join(settings.output_dir, "norm_test_predictions_ST.zarr"))

# Get evaluation metrics for all single task models and format for plotting
pearson_ST_df, spearman_ST_df = rnacomplete_metrics_sdata_table(sdata_test, b_presence_absence, target_cols, verbose=False, num_kmers=number_kmers, preds_suffix="_predictions_ST")
pearson_ST_long = pearson_ST_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_ST_long = spearman_ST_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_ST_long["Model"] = "SingleTask"
spearman_ST_long["Model"] = "SingleTask"
pearson_ST_long.to_csv(os.path.join(settings.output_dir, f"pearson_performance_{number_kmers}kmers_ST.tsv"), index=False, sep="\t")
spearman_ST_long.to_csv(os.path.join(settings.output_dir, f"spearman_performance_{number_kmers}kmers_ST.tsv"), index=False, sep="\t")

# Plot just the single task model eval
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_ST_long, x="Metric", y="Pearson", color="red", ax=ax[0])
sns.boxplot(data=spearman_ST_long, x="Metric", y="Spearman", color="red", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(settings.figure_dir, f"correlation_boxplots_{number_kmers}kmers_ST.pdf"))
