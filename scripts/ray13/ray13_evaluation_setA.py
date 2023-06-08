# General imports
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# EUGENe imports and settings
import eugene as eu
from eugene import settings
settings.dataset_dir = "/cellar/users/aklie/data/eugene/revision/ray13"
settings.output_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/output/revision/ray13"
settings.figure_dir = "/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/figures/revision/ray13"

# EUGENe packages
import seqdata as sd

# ray13 helpers
sys.path.append("/cellar/users/aklie/projects/ML4GLand/EUGENe_paper/scripts/ray13")
from ray13_helpers import rnacomplete_metrics

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

# Load in the Set A presence/absence predictions and make sure we have a good list of kmers
a_presence_absence = np.load(os.path.join(settings.dataset_dir, "setA_binary_ST.npy"))
setA_observed = sd.open_zarr(os.path.join(settings.dataset_dir, "norm_setA_ST.zarr"))[target_cols]
if number_kmers is not None:
    random_kmers = np.random.choice(np.arange(a_presence_absence.shape[0]), size=number_kmers)
    a_presence_absence = a_presence_absence[random_kmers, :]
    b_presence_absence = b_presence_absence[random_kmers, :]
valid_kmers = np.where((np.sum(a_presence_absence, axis=1) > 0) & (np.sum(b_presence_absence, axis=1) > 0))[0]
a_presence_absence = a_presence_absence[valid_kmers, :]
b_presence_absence = b_presence_absence[valid_kmers, :]

# Performing the above calculation for all targets
from scipy.stats import pearsonr, spearmanr
pearson_setA_long = pd.DataFrame()
spearman_setA_long = pd.DataFrame()
for i, task in tqdm(enumerate(target_cols), desc="Calculating metrics on each task", total=len(target_cols)):
    a_zscores, a_aucs, a_escores  = rnacomplete_metrics(a_presence_absence, setA_observed[task].values, verbose=False)
    b_zscores, b_aucs, b_escores = rnacomplete_metrics(b_presence_absence, setB_observed[task].values, verbose=False)
    try:
        zscore_nan_mask = np.isnan(a_zscores) | np.isnan(b_zscores)
        a_zscores = a_zscores[~zscore_nan_mask]
        b_zscores = b_zscores[~zscore_nan_mask]
        if len(a_zscores) > 0 and len(b_zscores) > 0:
            pearson_setA_long = pearson_setA_long.append(pd.Series({"RBP": task, "Metric": "Z-score", "Pearson": pearsonr(a_zscores, b_zscores)[0]}), ignore_index=True)
            spearman_setA_long = spearman_setA_long.append(pd.Series({"RBP": task, "Metric": "Z-score", "Spearman": spearmanr(a_zscores, b_zscores)[0]}), ignore_index=True)

        auc_nan_mask = np.isnan(a_aucs) | np.isnan(b_aucs)
        a_aucs = a_aucs[~auc_nan_mask]
        b_aucs = b_aucs[~auc_nan_mask]
        if len(a_aucs) > 0 and len(b_aucs) > 0:
            pearson_setA_long = pearson_setA_long.append(pd.Series({"RBP": task, "Metric": "AUC", "Pearson": pearsonr(a_aucs, b_aucs)[0]}), ignore_index=True)
            spearman_setA_long = spearman_setA_long.append(pd.Series({"RBP": task, "Metric": "AUC", "Spearman": spearmanr(a_aucs, b_aucs)[0]}), ignore_index=True)

        escore_nan_mask = np.isnan(a_escores) | np.isnan(b_escores)
        a_escores = a_escores[~escore_nan_mask]
        b_escores = b_escores[~escore_nan_mask]
        if len(a_escores) > 0 and len(b_escores) > 0:
            pearson_setA_long = pearson_setA_long.append(pd.Series({"RBP": task, "Metric": "E-score", "Pearson": pearsonr(a_escores, b_escores)[0]}), ignore_index=True)
            spearman_setA_long = spearman_setA_long.append(pd.Series({"RBP": task, "Metric": "E-score", "Spearman": spearmanr(a_escores, b_escores)[0]}), ignore_index=True)
    
    except:
        print(f"Could not evaluate {task}, skipping")
        continue

pearson_setA_long["Model"] = "SetA"
spearman_setA_long["Model"] = "SetA"
pearson_setA_long.to_csv(os.path.join(settings.output_dir, f"pearson_performance_{number_kmers}kmers_setA.tsv"), index=False, sep="\t")
spearman_setA_long.to_csv(os.path.join(settings.output_dir, f"spearman_performance_{number_kmers}kmers_setA.tsv"), index=False, sep="\t")

# Plot just the SetA results 
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_setA_long, x="Metric", y="Pearson", color="green", ax=ax[0])
sns.boxplot(data=spearman_setA_long, x="Metric", y="Spearman", color="green", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(settings.figure_dir, f"correlation_boxplots_{number_kmers}kmers_setA.pdf"))
