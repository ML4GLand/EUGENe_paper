import os
import yaml
import importlib
from eugene import settings, models

import torch
import torch.nn as nn
import torch.nn.functional as F
import eugene.models.base._layers as layers
import eugene.models.base._blocks as blocks
import eugene.models.base._towers as towers


def load_checkpoint_from_arch_config(
    ckpt_path,
    config_path,
    arch_name
):
    # If config path is just a filename, assume it's in the default config directory
    if "/" not in config_path:
        config_path = os.path.join(settings.config_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_type = getattr(importlib.import_module("kopp21_helpers"), arch_name)
    arch = model_type(**config["arch"])
    model = models.SequenceModule.load_from_checkpoint(ckpt_path, arch=arch)
    return model

class Kopp21CNN(nn.Module):
    """
    Custom convolutional model used in Kopp et al. 2021 paper

    PyTorch implementation of the TensorFlow model described here:
    https://github.com/wkopp/janggu_usecases/tree/master/01_jund_prediction

    This model can only be run in "ds" mode. The reverse complement must be included in the Dataloader
    Parameters
    ----------
    input_len : int
        Length of the input sequence.
    output_dim : int
        Dimension of the output.
    strand : str, optional
        Strand of the input. This model is only implemented for "ds"
    task : str, optional
        Task for this model. By default "binary_classification" for this mode
    aggr : str, optional
        Aggregation method. Either "concat", "max", or "avg". By default "max" for this model.
    filters : list, optional
        Number of filters in the convolutional layers. 
    conv_kernel_size : list, optional
        Kernel size of the convolutional layers.
    maxpool_kernel_size : int, optional
        Kernel size of the maxpooling layer.
    stride : int, optional
        Stride of the convolutional layers.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        aggr: str = "max",
        filters: list = [10, 8],
        conv_kernel_size: list = [11, 3],
        maxpool_kernel_size: int = 30,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ):
        super(Kopp21CNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.aggr = aggr
        self.revcomp = layers.RevComp()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv = nn.Conv1d(4, filters[0], conv_kernel_size[0], stride=stride)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm1d(filters[0])
        self.conv2 = nn.Conv1d(filters[0], filters[1], conv_kernel_size[1], stride=stride)
        self.relu2 = nn.ReLU(inplace=False)        
        self.batchnorm2 = nn.BatchNorm1d(filters[1])
        self.linear = nn.Linear(filters[1], self.output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x_rev_comp = self.revcomp(x)
        x_fwd = self.conv(x)
        x_fwd = self.relu(x_fwd)
        x_rev_comp = self.conv(x_rev_comp)
        x_rev_comp = self.relu(x_rev_comp)
        if self.aggr == "concat":
            x = torch.cat((x_fwd, x_rev_comp), dim=2)
        elif self.aggr == "max":
            x = torch.max(x_fwd, x_rev_comp)
        elif self.aggr == "avg":
            x = (x_fwd + x_rev_comp) / 2
        elif self.aggr is None:
            x = torch.cat((x_fwd, x_rev_comp), dim=1)
        x = self.maxpool(x)
        x = self.batchnorm(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool1d(x, x.shape[2])
        x = self.batchnorm2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
    