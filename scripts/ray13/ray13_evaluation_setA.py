import os
import glob
import logging
import torch
import numpy as np
import pandas as pd
import eugene as eu
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

eu.settings.dataset_dir = "/cellar/users/aklie/data/eugene/ray13"
eu.settings.output_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/output/ray13"
eu.settings.logging_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/logs/ray13"
eu.settings.config_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/configs/ray13"
figure_dir = "/cellar/users/aklie/projects/EUGENe/EUGENe_paper/figures/ray13"
eu.settings.verbosity = logging.ERROR
number_kmers=None

# Load the test data
sdata_test = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, "norm_setB_processed_ST.h5sd"))
target_mask = sdata_test.seqs_annot.columns.str.contains("RNCMPT")
target_cols = sdata_test.seqs_annot.columns[target_mask]

# Load in the Set B presence/absence predictions
b_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "setB_binary.npy"))
setB_observed = sdata_test.seqs_annot[target_cols]

###################
# Set A performance
###################
# Load in the Set A presence/absence predictions
a_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "setA_binary_ST.npy"))
setA_observed = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, eu.settings.dataset_dir, "norm_setA_processed_ST.h5sd")).seqs_annot[target_cols]

# Performing the above calculation for all targets (TODO: parallelize and simplify)
from scipy.stats import pearsonr, spearmanr
pearson_setA_long = pd.DataFrame()
spearman_setA_long = pd.DataFrame()
for i, task in tqdm(enumerate(target_cols), desc="Calcualting metrics on each task", total=len(target_cols)):
    a_zscores, a_aucs, a_escores  = eu.predict.rna_complete_metrics_apply(a_presence_absence, setA_observed[task].values, verbose=False, use_calc_auc=True, num_kmers=number_kmers)
    b_zscores, b_aucs, b_escores = eu.predict.rna_complete_metrics_apply(b_presence_absence, setB_observed[task].values, verbose=False, use_calc_auc=True, num_kmers=number_kmers) 
    
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

pearson_setA_long["Model"] = "SetA"
pearson_setA_long["Model"] = "SetA"
pearson_setA_long.to_csv(os.path.join(eu.settings.output_dir, "pearson_performance_setA.tsv"), index=False, sep="\t")
pearson_setA_long.to_csv(os.path.join(eu.settings.output_dir, "spearman_performance_setA.tsv"), index=False, sep="\t")

# Plot just the SetA results 
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_setA_long, x="Metric", y="Pearson", color="green", ax=ax[0])
sns.boxplot(data=spearman_setA_long, x="Metric", y="Spearman", color="green", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "correlation_boxplots_setA.pdf"))
