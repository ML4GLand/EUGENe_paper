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
number_kmers=100

# Load the test data
sdata_test = eu.dl.read_h5sd(os.path.join(eu.settings.dataset_dir, "norm_setB_processed_ST.h5sd"))
target_mask = sdata_test.seqs_annot.columns.str.contains("RNCMPT")
target_cols = sdata_test.seqs_annot.columns[target_mask]

# Load in the Set B presence/absence predictions
b_presence_absence = np.load(os.path.join(eu.settings.dataset_dir, "setB_binary.npy"))
setB_observed = sdata_test.seqs_annot[target_cols]

#########################
# Kipoi model performance
#########################

# We need to get the protein IDs from the motifs in the
id_mapping = pd.read_excel(os.path.join(eu.settings.dataset_dir, "hg19_motif_hits", "ID.mapping.xls"), sheet_name=0)
id_mp = id_mapping.set_index("Motif ID")["Protein(s)"]
cols_w_ids = ~target_cols.map(id_mp).isna()
target_cols_w_ids = target_cols[cols_w_ids]
ids_w_target_cols = pd.Index([id.split("(")[0].rstrip() for id in target_cols_w_ids.map(id_mp)])

# Get the kipoi models names
db_model_names = eu.external.kipoi.get_model_names("DeepBind/Homo_sapiens/RBP/D")

# Get predictions with each model and store them in sdata
target_cols_w_model = []
for i, (protein_id , motif_id) in tqdm(enumerate(zip(ids_w_target_cols, target_cols_w_ids)), desc="Importing models", total=len(ids_w_target_cols)):
    print("Predicting for protein: ", protein_id, " motif: ", motif_id)
    db_model_name = db_model_names[db_model_names.str.contains(protein_id)]
    if len(db_model_name) == 0:
        print("No model found for protein: ", protein_id)
        continue
    try:
        model = eu.external.kipoi.get_model(db_model_name.values[0])
        sdata_test[f"{motif_id}_predictions_kipoi"] = model(sdata_test.ohe_seqs).cpu().numpy()
        target_cols_w_model.append(motif_id)
    except:
        print("Failed to load model")

################
# Saving results
################
                
# Save the sdata with kipoi predictions
sdata_test.write_h5sd(os.path.join(eu.settings.output_dir, "norm_test_predictions_kipoi.h5sd"))

# Evaluate the predictions using the RNAcompete metrics
pearson_kipoi_df, spearman_kipoi_df = eu.predict.rnacomplete_metrics_sdata_table(sdata_test, b_presence_absence, target_cols_w_model, verbose=False, num_kmers=number_kmers, preds_suffix="_predictions_kipoi")
pearson_kipoi_long = pearson_kipoi_df.reset_index().melt(id_vars="index", value_name="Pearson", var_name="Metric").rename({"index":"RBP"}, axis=1)
spearman_kipoi_long = spearman_kipoi_df.reset_index().melt(id_vars="index", value_name="Spearman", var_name="Metric").rename({"index":"RBP"}, axis=1)
pearson_kipoi_long["Model"] = "Kipoi"
spearman_kipoi_long["Model"] = "Kipoi"
pearson_kipoi_long.to_csv(os.path.join(eu.settings.output_dir, f"pearson_performance_{number_kmers}kmers_kipoi.tsv"), index=False, sep="\t")
spearman_kipoi_long.to_csv(os.path.join(eu.settings.output_dir, f"spearman_performance_{number_kmers}kmers_kipoi.tsv"), index=False, sep="\t")

# Plot just the kipoi results as boxplots
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(data=pearson_kipoi_long, x="Metric", y="Pearson", color="orange", ax=ax[0])
sns.boxplot(data=spearman_kipoi_long, x="Metric", y="Spearman", color="orange", ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, f"correlation_boxplots_{number_kmers}kmers_kipoi.pdf"))
