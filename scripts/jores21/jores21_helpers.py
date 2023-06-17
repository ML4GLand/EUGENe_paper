import torch
import torch.nn as nn
import torch.nn.functional as F 
from eugene.models.base import _layers as layers
from eugene.models.base import _blocks as blocks
from eugene.models.base import _towers as towers


class BiConv1DTower(nn.Module):
    def __init__(
        self, 
        filters: int,
        kernel_size: int,
        input_size: int = 4, 
        n_layers: int = 1, 
        stride: int = 1, 
        dropout_rate: float = 0.15
    ):
        super().__init__()

        # Set-up attributes of model
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_size = input_size
        if n_layers < 1:
            raise ValueError("At least one layer needed")
        self.n_layers = n_layers
        if (dropout_rate < 0) or (dropout_rate > 1):
            raise ValueError("Dropout rate must be a float between 0 and 1")
        self.dropout_rate = dropout_rate
        self.stride = stride

        # Set-up the lists for layers
        self.kernels = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.relus = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Start with a kernel
        kernel = nn.Parameter(torch.zeros(self.filters, self.input_size, kernel_size))
        nn.init.xavier_uniform_(kernel)
        self.kernels.append(kernel)
        
        # Add a bias
        bias = nn.Parameter(torch.zeros(filters))
        nn.init.zeros_(bias)
        self.biases.append(bias)

        # Add a relu
        relu = nn.ReLU(inplace=False)
        self.relus.append(relu)

        # Add a dropout
        dropout = nn.Dropout(p=self.dropout_rate)
        self.dropouts.append(dropout)

        # Add the rest of the layers
        for _ in range(1, self.n_layers):
            kernel = nn.Parameter(torch.empty((self.filters, self.filters, self.kernel_size)))
            nn.init.xavier_uniform_(kernel)
            self.kernels.append(kernel)
            bias = nn.Parameter(torch.empty((self.filters)))
            nn.init.zeros_(bias)
            self.biases.append(bias)
            self.relus.append(nn.ReLU(inplace=False))
            self.dropouts.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, x):
        x_fwd = F.conv1d(x, self.kernels[0], stride=self.stride, padding="same")
        x_fwd = torch.add(x_fwd.transpose(1, 2), self.biases[0]).transpose(1, 2)
        x_fwd = self.dropouts[0](self.relus[0](x_fwd))
        x_rev = F.conv1d(x, torch.flip(self.kernels[0], dims=[0, 1]), stride=self.stride, padding="same")
        x_rev = torch.add(x_rev.transpose(1, 2), self.biases[0]).transpose(1, 2)
        x_rev = self.dropouts[0](self.relus[0](x_rev))
        for i in range(1, self.n_layers):
            x_fwd = F.conv1d(x_fwd, self.kernels[i], stride=self.stride, padding="same")
            x_fwd = torch.add(x_fwd.transpose(1, 2), self.biases[i]).transpose(1, 2)
            x_fwd = self.dropouts[i](self.relus[i](x_fwd))
            x_rev = F.conv1d(x_rev, torch.flip(self.kernels[i], dims=[0, 1]), stride=self.stride, padding="same")
            x_rev = torch.add(x_rev.transpose(1, 2), self.biases[i]).transpose(1, 2)
            x_rev = self.dropouts[i](self.relus[i](x_rev))
        return torch.add(x_fwd, x_rev)

class Jores21CNN(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        filters: int = 128,
        kernel_size: int = 13,
        layers: int = 2,
        stride: int = 1,
        dropout: float = 0.15,
        hidden_dim: int = 64,
    ):
        super(Jores21CNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.layers = layers
        self.stride = stride
        self.dropout = dropout

        # Create the blocks
        self.biconv = BiConv1DTower(
            filters=filters,
            kernel_size=kernel_size,
            n_layers=layers,
            stride=stride,
            dropout_rate=dropout,
        )
        self.conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=input_len * filters, out_features=hidden_dim)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.biconv(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.batchnorm(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x
    