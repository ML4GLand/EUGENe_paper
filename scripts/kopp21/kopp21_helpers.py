import torch
import torch.nn as nn
import eugene.models.base._layers as layers
import eugene.models.base._blocks as blocks
import eugene.models.base._towers as towers


class dsFCN(nn.Module):
    """
    Instantiate a fully connected neural network with the specified layers and parameters.
    
    By default, this architecture flattens the one-hot encoded sequence and passes 
    it through a set of layers that are fully connected. The task defines how the output is
    treated (e.g. sigmoid activation for binary classification). The loss function is
    should be matched to the task (e.g. binary cross entropy ("bce") for binary classification).

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        input_dims: int = 4,
        aggr: str = "concat",
        dense_kwargs: dict = {}
    ):
        super(dsFCN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.dense_kwargs = dense_kwargs
        self.aggr = aggr
        self.flattened_input_dims = input_len * input_dims

        # Creat the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the block
        if aggr == "concat":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims * 2,
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif aggr in ["max", "avg"]:
            self.dense_block = blocks.DenseBlock(
                input_dim=self.flattened_input_dims,
                output_dim=output_dim, 
                **dense_kwargs
            )
        else:
            raise ValueError(f"Invalid aggr, must be one of ['concat', 'max', 'avg'], got {aggr}.")

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        x = x.flatten(start_dim=1)
        x_rev_comp = x_rev_comp.flatten(start_dim=1)
        if self.aggr == "concat":
            x = torch.cat([x, x_rev_comp], dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x
    

class dsCNN(nn.Module):
    """
    Instantiate a CNN model with a set of convolutional layers and a set of fully
    connected layers.

    By default, this architecture passes the one-hot encoded sequence through a set
    1D convolutions with 4 channels. The task defines how the output is treated (e.g.
    sigmoid activation for binary classification). The loss function is should be matched
    to the task (e.g. binary cross entropy ("bce") for binary classification).

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the flattened output of the convolutional layers directly to 
        the output layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        aggr: str = "concat",
        dense_kwargs: dict = {},
    ):
        super(dsCNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs
        self.aggr = aggr

        # Create the revcomp layer
        self.revcomp = layers.RevComp()
        
        # Create the conv1d tower
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            **conv_kwargs
        )
        
        # Get the dimension of the output of the conv1d tower
        if aggr == "concat":
            self.dense_dim = self.conv1d_tower.flatten_dim * 2
        elif aggr in ["max", "avg"]:
            self.dense_dim = self.conv1d_tower.flatten_dim
        else:
            raise ValueError("aggr must be one of ['concat', 'max', 'avg']")
        
        # Create the dense block
        self.dense_block = blocks.DenseBlock(
            input_dim=self.dense_dim, 
            output_dim=output_dim, 
            **dense_kwargs
        )

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        
        x = self.conv1d_tower(x)
        x = x.view(x.size(0), self.conv1d_tower.flatten_dim)

        x_rev_comp = self.conv1d_tower(x_rev_comp)
        x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.conv1d_tower.flatten_dim)

        if self.aggr == "concat":
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x
    

class dsRNN(nn.Module):
    """
    Instantiate an RNN model with a set of recurrent layers and a set of fully
    connected layers.

    By default, this model passes the one-hot encoded sequence through recurrent layers
    and then through a set of fully connected layers. The output of the fully connected
    layers is passed to the output layer.

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer. If not provided, the
        default passes the recurrent output of the recurrent layers directly to the
        output layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        recurrent_kwargs: dict,
        aggr: str = "concat",
        input_dims: int = 4,
        dense_kwargs: dict = {},
    ): 
        super(dsRNN, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs

        # Create the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the recurrent block
        self.recurrent_block = blocks.RecurrentBlock(
            input_dim=input_dims,
            **recurrent_kwargs
        )

        # Create the dense block
        if aggr == "concat":
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels * 2,
                output_dim=output_dim, 
                **dense_kwargs
        )
        elif aggr in ["max", "avg"]:
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        
    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        x, _ = self.recurrent_block(x)
        x = x[:, -1, :]
        x_rev_comp, _ = self.recurrent_block(x_rev_comp)
        x_rev_comp = x_rev_comp[:, -1, :]
        if self.aggr == "concat":
            x = torch.cat((x, x_rev_comp), dim=1)
            x = self.dense_block(x)
        elif self.aggr in ["max", "avg"]:
            x = self.dense_block(x)
            x_rev_comp = self.dense_block(x_rev_comp)
            if self.aggr == "max":
                x = torch.max(x, x_rev_comp)
            elif self.aggr == "avg":
                x = (x + x_rev_comp) / 2
        return x


class dsHybrid(nn.Module):
    """
    A hybrid model that uses both a CNN and an RNN to extract features then passes the
    features through a set of fully connected layers.
    
    By default, the CNN is used to extract features from the input sequence, and the RNN is used to 
    to combine those features. The output of the RNN is passed to a set of fully connected
    layers to make the final prediction.

    Parameters
    ----------
    input_len:
        The length of the input sequence.
    output_dim:
        The dimension of the output.
    task:
        The task of the model.
    dense_kwargs:
        The keyword arguments for the fully connected layer.
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        conv_kwargs: dict,
        recurrent_kwargs: dict,
        aggr="concat_cnn",
        dense_kwargs: dict = {},
    ):
        super(dsHybrid, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim
        self.conv_kwargs = conv_kwargs
        self.recurrent_kwargs = recurrent_kwargs
        self.dense_kwargs = dense_kwargs
        self.aggr = aggr

        # Create the revcomp layer
        self.revcomp = layers.RevComp()

        # Create the conv1d tower
        self.conv1d_tower = towers.Conv1DTower(
            input_len=input_len,
            **conv_kwargs
        )

        # Create the recurrent block and dense block
        if aggr == "concat_cnn":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels * 2, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
        )
        elif aggr == "concat_rnn":
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels*2, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        elif aggr in ["max", "avg"]:
            self.recurrent_block = blocks.RecurrentBlock(
                input_dim=self.conv1d_tower.out_channels, 
                **recurrent_kwargs
            )
            self.dense_block = blocks.DenseBlock(
                input_dim=self.recurrent_block.out_channels, 
                output_dim=output_dim, 
                **dense_kwargs
            )
        else:
            raise ValueError("aggr must be one of ['concat_cnn', 'concat_rnn', 'max', 'avg']")

    def forward(self, x):
        x_rev_comp = self.revcomp(x)
        
        x = self.conv1d_tower(x)
        x = x.transpose(1, 2)

        x_rev_comp = self.conv1d_tower(x_rev_comp)
        x_rev_comp = x_rev_comp.transpose(1, 2)

        if self.aggr == "concat_cnn":
            x = torch.cat((x, x_rev_comp), dim=2)
            out, _ = self.recurrent_block(x)
            out = self.dense_block(out[:, -1, :])
        elif self.aggr == "concat_rnn":
            out, _ = self.recurrent_block(x)
            out_rev_comp, _ = self.recurrent_block(x_rev_comp)
            out = torch.cat((out[:, -1, :], out_rev_comp[:, -1, :]), dim=1)
            out = self.dense_block(out)
        elif self.aggr in ["max", "avg"]:
            out_, _ = self.recurrent_block(x)
            out = self.dense_block(out_[:, -1, :])
            out_rev_comp, _ = self.recurrent_block(x_rev_comp)
            out_rev_comp = self.dense_block(out_rev_comp[:, -1, :])
            if self.aggr == "max":
                out = torch.max(out, out_rev_comp)
            elif self.aggr == "avg":
                out = (out + out_rev_comp) / 2
        return out