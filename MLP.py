"""
Multi-Layer Perceptron
----------------------

Implementation of a fully-connected neural network.

An example usage is as a main model, that doesn't include any trainable weights.
Instead, weights are received as additional inputs. For instance, using an
auxilliary network, a so called hypernetwork, see

    Ha et al., "HyperNetworks", arXiv, 2016,
    https://arxiv.org/abs/1609.09106
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PE import PositionalEncoder


class MLP(nn.Module):
    """Implementation of a Multi-Layer Perceptron (MLP).
       Needs to take in weights from an HN (hypernetwork).
       Adapted from: https://github.com/chrhenning/hypnettorch

    Args:
        activation_fn: The nonlinearity used in hidden layers. If ``None``, no
                       nonlinearity will be applied.
        use_batch_norm (bool): Whether batch normalization should be used. Will
                               be applied before the activation function in all hidden layers.
        out_fn (optional): If provided, this function will be applied to the
                           output neurons of the network.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
    """

    def __init__(self,
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 out_fn=None,
                 verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)

        self._dropout_rate = 0.0  # dropout rate: float values (default set to 0.0)
        self._dropout = F.dropout
        self._use_batch_norm = use_batch_norm  # if True, use batch norm
        self._out_fn = out_fn  # out function, nn.Module
        self._a_fn = activation_fn

    def forward(self,
                x,
                h_container,
                weights):
        """Compute the output :math:`y` of this network given the input :math:`x`.
        Args:
            x: torch.floattensor
            h_container: torch.float_tensor
            weights: list of torch.tensor
        Returns:
            (tuple): Tuple containing:
            - **y**: The output of the network.
            - **h_y** (optional): If ``out_fn`` was specified in the
              constructor, then this value will be returned.
        """

        if weights is None:
            raise Exception('Forward computation needs to take in weights')
        # process the weights and bias into a list
        W_weights = []
        b_weights = []
        for l in range(len(weights)):
            if l % 2 == 0:
                W_weights.append(weights[l])
            else:
                b_weights.append(weights[l])

        hidden = x
        for l in range(len(W_weights)):
            W = W_weights[l]
            b = b_weights[l]

            # Linear layer.
            hidden = F.linear(hidden, W, bias=b)
            # Only for hidden layers.
            if l < len(W_weights) - 1:
                # Batch norm
                if self._use_batch_norm:
                    batch_mean = torch.mean(hidden, dim=0).detach()
                    batch_var = torch.var(hidden, dim=0).detach()
                    hidden = F.batch_norm(hidden, batch_mean, batch_var, weight=None, bias=None, training=False)
                # Dropout
                hidden = self._dropout(hidden, p=h_container['dropout'])
                # Non-linearity
                if self._a_fn is not None:
                    hidden = self._a_fn(hidden)
        if self._out_fn is not None:
            return self._out_fn(hidden), hidden
        return hidden


class MLP_Hcontainer():
    def __init__(self,
                 hiddn,
                 pe_size,
                 pe_m,
                 dropout=0.0,
                 weight_decay=0.0,
                 _lambda=0.0,
                 device="cuda",
                 ):
        self.pe = PositionalEncoder(size=pe_size,
                                    m=pe_m)
        hiddn_lst = self.pe.arch_pos_emb(hiddn)
        self.H_container_dict = {"dropout": dropout,
                                 "weight_decay": weight_decay,
                                 "_lambda": _lambda,
                                 "hiddn_pe": hiddn_lst}
        self.H_container_dict = OrderedDict(sorted(self.H_container_dict.items()))
        self.device = device

    def __getitem__(self, key):
        return self.H_container_dict[key]

    def __setitem__(self, key, value):
        self.H_container_dict[key] = value

    def __repr__(self):
        return repr(self.H_container_dict)

    def to_torch_tensor(self):
        tensor_lst = []
        for x, y in self.H_container_dict.items():
            if type(y) != torch.Tensor:
                y = torch.Tensor([y])
            tensor_lst.append(y)
        ret_ = torch.cat(tensor_lst, dim=0).view(-1).type(torch.FloatTensor)
        ret_ = ret_.to(self.device)
        return ret_

    def set_H_container(self,
                        dropout=None,
                        weight_decay=None,
                        hiddn=None,
                        _lambda=None):
        if dropout is not None:
            self.H_container_dict["dropout"] = dropout
        if weight_decay is not None:
            self.H_container_dict["weight_decay"] = weight_decay
        if hiddn is not None:
            new_hiddn_lst = self.pe.arch_pos_emb(hiddn)
            self.H_container_dict["hidden_pe"] = torch.tensor(new_hiddn_lst)
        if _lambda is not None:
            self.H_container_dict["_lambda"] = _lambda

    def get_H_container(self):
        return self.H_container_dict


def MLP_weight_shapes(n_in=1,
                      n_out=1,
                      hidden_layers=[10, 10],
                      use_bias=True):
    """Compute the tensor shapes of all parameters in a fully-connected network.
    Args:
        n_in: Number of inputs.
        n_out: Number of output units.
        hidden_layers: A list of ints, each number denoting the size of a
            hidden layer.
        use_bias: Whether the FC layers should have biases.
    Returns:
        A list of list of integers, denoting the shapes of the individual
        parameter tensors.
    """
    shapes = []
    prev_dim = n_in
    layer_out_sizes = hidden_layers + [n_out]
    for i, size in enumerate(layer_out_sizes):
        shapes.append([size, prev_dim])
        if use_bias:
            shapes.append([size])
        prev_dim = size
    return shapes
