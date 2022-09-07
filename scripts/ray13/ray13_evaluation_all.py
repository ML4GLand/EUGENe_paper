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
number_kmers=10


# Load the test data
sdata_test = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, "norm_setB_processed_ST.h5sd"))
target_mask = sdata_test.seqs_annot.columns.str.contains("RNCMPT")
target_cols = sdata_test.seqs_annot.columns[target_mask]

# Load in the Set B presence/absence predictions
b_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "SetB_binary.npy"))
setB_observed = sdata_test.seqs_annot[target_cols]


###################
# Set A performance
###################
"""
# Load in the Set A presence/absence predictions
a_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "SetA_binary_ST.npy"))
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
"""

#########################
# Single task performance
#########################

# Get predictions on the test data from all single task models
trained_model_cols = []
for i, target_col in enumerate(target_cols):
    print(f"Testing DeepBind SingleTask model on {target_col}")
    try:
        model_file = glob.glob(os.path.join(eu.settings.logging_dir, "DeepBind_ST", target_col, "checkpoints", "*"))[0]
        model = eu.models.DeepBind.load_from_checkpoint(model_file)
        trained_model_cols.append(target_col)
    except:
        print(f"No model trained for {target_col}")
        continue
    eu.settings.dl_num_workers = 0
    eu.predict.predictions(
        model,
        sdata=sdata_test, 
        target=target_col,
        name="DeepBind_ST",
        version=target_col,
        file_label="test",
        suffix="_ST"
    )
    del model
    
# Get evaluation metrics for all single task models and format for plotting
pearson_ST_df, spearman_ST_df = eu.predict.summarize_rbps_apply(sdata_test, b_presence_absence, trained_model_cols, use_calc_auc=True, verbose=False, n_kmers=number_kmers, preds_suffix="_predictions_ST")
pearson_ST_long = pearson_ST_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_ST_long = spearman_ST_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_ST_long["Model"] = "SingleTask"
spearman_ST_long["Model"] = "SingleTask"
pearson_ST_long.to_csv(os.path.join(eu.settings.output_dir, "pearson_performance_ST.tsv"), index=False, sep="\t")
spearman_ST_long.to_csv(os.path.join(eu.settings.output_dir, "spearman_performance.tsv_ST"), index=False, sep="\t")

# Plot just the single task model eval
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_ST_long, x="Metric", y="Pearson", color="red", ax=ax[0])
sns.boxplot(data=spearman_ST_long, x="Metric", y="Spearman", color="red", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "correlation_boxplots_ST.pdf"))


#######################
# Multitask performance
#######################

# Also need the multi-task columns (single task we could train on all the columns)
sdata_training = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, eu.settings.dataset_dir, "norm_setA_sub_MT.h5sd"))
target_mask_MT = sdata_training.seqs_annot.columns.str.contains("RNCMPT")
target_cols_MT = sdata_training.seqs_annot.columns[target_mask_MT]
del sdata_training
len(target_cols_MT)

# Get predictions on the test data from all multi task models
print("Testing DeepBind MultiTask model on")
version = 0
model_file = glob.glob(os.path.join(eu.settings.logging_dir, "DeepBind_MT", f"v{version}", "checkpoints", "*"))[0]
model = eu.models.DeepBind.load_from_checkpoint(model_file)
eu.settings.dl_num_workers = 0
eu.predict.predictions(
    model,
    sdata=sdata_test, 
    target=target_cols_MT,
    name="DeepBind_MT",
    version=f"v{version}",
    file_label="test",
    suffix="_MT"
)
del model

# Get evaluation metrics for all single task models and format for plotting
pearson_MT_df, spearman_MT_df = eu.predict.summarize_rbps_apply(sdata_test, b_presence_absence, target_cols_MT, use_calc_auc=True, verbose=False, n_kmers=number_kmers, preds_suffix="_predictions_MT")
pearson_MT_long = pearson_MT_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_MT_long = spearman_MT_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_MT_long["Model"] = "MultiTask"
spearman_MT_long["Model"] = "MultiTask"
pearson_MT_long.to_csv(os.path.join(eu.settings.output_dir, "pearson_performance_MT.tsv"), index=False, sep="\t")
spearman_MT_long.to_csv(os.path.join(eu.settings.output_dir, "spearman_performance.tsv_MT"), index=False, sep="\t")

# Plot just the multi task model eval
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_MT_long, x="Metric", y="Pearson", color="blue", ax=ax[0])
sns.boxplot(data=spearman_MT_long, x="Metric", y="Spearman", color="blue", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "correlation_boxplots_MT.pdf"))

################
# Saving results
################

# Save the sdata with the predictions for single task, and multitask models
sdata_test.write_h5sd(os.path.join(eu.settings.output_dir, "norm_test_predictions_ST_MT.h5sd"))

# Concatenate and save the dataframes for plotting across models
#pearson_long = pd.concat([pearson_setA_long, pearson_MT_long, pearson_ST_long])
#spearman_long = pd.concat([spearman_setA_long, spearman_MT_long, spearman_ST_long])
pearson_long = pd.concat([pearson_MT_long, pearson_ST_long])
spearman_long = pd.concat([spearman_MT_long, spearman_ST_long])
pearson_models = pearson_long.pivot(index=["RBP", "Metric"], columns="Model", values="Pearson").reset_index()
spearman_models = spearman_long.pivot(index=["RBP", "Metric"], columns="Model", values="Spearman").reset_index()
pearson_models.to_csv(os.path.join(eu.settings.output_dir, "pearson_performance.tsv"), index=False, sep="\t")
spearman_models.to_csv(os.path.join(eu.settings.output_dir, "spearman_performance.tsv"), index=False, sep="\t")
