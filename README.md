# Welcome to the EUGENe Paper Repo
This repository contains the code used to generate the results presented in the manuscript "EUGENe: A Python toolkit for predictive analyses of regulatory sequences."

Each directory within this repository is broken up into the three use cases (section titles in *italics*) presented in the above preprint:
1. `jores21` -- *STARR-seq plant promoter activity prediction*
2. `ray13` -- *In vitro RNA binding prediction with DeepBind*
3. `kopp21` -- *JunD ChIP-seq binding classification*

# EUGENe installation
You can install the version of EUGENe used for the preprint with `pip`

```bash
pip install eugene-tools==0.1.2
```

# Datasets
You can find the raw and processed data for running the code and generating results at the following Zenodo link: https://zenodo.org/deposit/7140083#.

## Subdirectories

## `configs`
These contain `.yaml` files used for training models in each use case (when applicable).

## `notebooks`
Notebooks for each use case are organized as follows:

- `dataset_ETL.ipynb` — extract the data from it’s downloaded or raw format, transform and preprocess it through a series of steps and get it ready for loading into a PyTorch model
- `dataset_EDA.ipynb` — perform visualizations of your data to better understand what is going on with it. You can often iterate between this and ETL to get a final version of your data for loading into a model
- `dataset_training.ipynb` — train a single or multiple models on one or multiple iterations of the dataset. This notebook or section is reserved for calls to fit and visualizations of training summaries
- `dataset_evaluate.ipynb` — evaluate trained models on test data and visualize and summarize the performance. This often starts with loading in the best iteration of the model from the training notebook and getting predictions on some test data of interest. Once predictions are generated, they can be added to SeqData or loaded in to generate useful summaries and visualizations
- `dataset_intepret.ipynb` — interpret trained models with either test data or random data that is manipulated by model outputs or prior knowledge. This can often be combined with the previous notebook, but can sometimes be standalone
- `dataset_plotting.ipynb` - generate plots not already generated in the previous notebooks.
- `dataset_gpu_util.ipynb` — tests to make sure EUGENe is using the GPU and that the GPU is working properly

There is also the `plotting.ipynb` notebook in the `training_mem` folder where we show the results of using SeqData to load large datasets out-of-core!

**Note**: If you want to compare the DeepBind models to Kipoi's submitted DeepBind models, you will need to install Kipoi: https://github.com/kipoi/kipoi

## `scripts`
These contain Python scripts for when you have to submit a job to a cluster or run it on your local machine behind a screen because it will take too long otherwise. These are organized in a similar manner to the `notebooks` for each use case.
