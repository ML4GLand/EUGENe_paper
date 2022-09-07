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
b_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "setB_binary.npy"))
setB_observed = sdata_test.seqs_annot[target_cols]

#########################
# Single task performance
#########################

# Get predictions on the test data from all single task models
trained_model_cols = []
for i, target_col in enumerate(target_cols[:7]):
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
pearson_ST_df, spearman_ST_df = eu.predict.summarize_rbps_apply(sdata_test, b_presence_absence, trained_model_cols[:7], use_calc_auc=True, verbose=False, n_kmers=number_kmers, preds_suffix="_predictions_ST")
pearson_ST_long = pearson_ST_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_ST_long = spearman_ST_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_ST_long["Model"] = "SingleTask"
spearman_ST_long["Model"] = "SingleTask"
pearson_ST_long.to_csv(os.path.join(eu.settings.output_dir, "pearson_performance_ST.tsv"), index=False, sep="\t")
spearman_ST_long.to_csv(os.path.join(eu.settings.output_dir, "spearman_performance_ST.tsv"), index=False, sep="\t")

# Plot just the single task model eval
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_ST_long, x="Metric", y="Pearson", color="red", ax=ax[0])
sns.boxplot(data=spearman_ST_long, x="Metric", y="Spearman", color="red", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "correlation_boxplots_ST.pdf"))

################
# Saving results
################

# Save the sdata with the predictions for single task, and multitask models
sdata_test.write_h5sd(os.path.join(eu.settings.output_dir, "norm_test_predictions_ST.h5sd"))