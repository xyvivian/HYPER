# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :utils/torch_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/11/2019
# @version        :1.0
# @python_version :3.6.8


"""
A collection of helper functions that should capture common functionalities
needed when working with PyTorch.
"""
import math

import numpy as np
import torch
from torch import nn


def init_params(weights, bias=None):
    """Initialize the weights and biases of a linear or (transpose) conv layer.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    Args:
        weights: The weight tensor to be initialized.
        bias (optional): The bias tensor to be initialized.
    """
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


def generate_input_dim_lst(num_layer=4,
                           max_layer=8,
                           compression=2,
                           input_dim=12,
                           threshold=1):
    """
    Params: num_layer: number of layers in the neural network
            compression: the ratio for input layer decaying
            input_dim: dimension of the data
            threshold: the minimum number of neurons per layer
    """
    from copy import deepcopy
    assert (max_layer % 2 == 0)
    assert (num_layer % 2 == 0)  # assert both be divisible by 2-> for an hourglass structure
    input_dim_list = []
    for i in range(int(num_layer / 2)):
        if int(input_dim / (compression ** (i + 1))) >= threshold:
            input_dim_list.append(int(input_dim / (compression ** (i + 1))))
        else:
            input_dim_list.append(threshold)
    zeros = [0] * (max_layer - num_layer)
    input_dim_list_reverse = input_dim_list[0:-1][::-1]
    input_dim_list.extend(zeros)
    input_dim_list_reverse.append(input_dim)
    input_dim_list.extend(input_dim_list_reverse)

    hidden_dim_list = deepcopy(input_dim_list)
    input_dim_list.insert(0, input_dim)

    # Fast computation of A matrix -> instead of building a tensor of shape
    # Max_layer x input_dim x input_dim,
    # We build a list to specify where the zeros are
    arch = []
    for i in range(1, len(input_dim_list)):
        last_nonzero = next((x for x in reversed(input_dim_list[0:i]) if x != 0), None)
        arch.append([input_dim_list[i], last_nonzero])
    return arch, input_dim_list, hidden_dim_list


def generate_architecture(input_dim_list):
    arch = []
    input_dim = input_dim_list[-1]
    for i in range(1, len(input_dim_list)):
        last_nonzero = next((x for x in reversed(input_dim_list[0:i]) if x != 0), None)
        arch.append([input_dim_list[i], last_nonzero])
    return arch


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y):
        super(PyODDataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        sample_y = self.y[idx]
        sample_torch = torch.from_numpy(sample)

        return sample_torch, sample_y, idx


def pyod_mlp_loader2(train_set, args):
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               worker_init_fn=np.random.seed(args.random_seed))
    return train_loader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sample_hps(decay_mu, decay_std, dropout_mu, dropout_std, wd_mu, wd_std, sample_size):
    sample_decay = np.random.normal(decay_mu, decay_std, size=sample_size)
    sample_dropout = np.random.normal(dropout_mu, dropout_std, size=sample_size)
    sample_wd = np.random.normal(wd_mu, wd_std, size=sample_size)

    sample_decay = sample_decay[sample_decay >= 1]
    sample_decay = sample_decay[sample_decay <= 3]

    sample_dropout = sample_dropout[sample_dropout >= 0]
    sample_dropout = sample_dropout[sample_dropout < 0.5]

    sample_wd = sample_wd[sample_wd >= 0]
    sample_wd = sample_wd[sample_wd <= 0.1]

    if len(sample_decay) == 0:
        sample_decay = np.asarray([decay_mu])

    if len(sample_dropout) == 0:
        sample_dropout = np.asarray([dropout_mu])

    if len(sample_wd) == 0:
        sample_wd = np.asarray([wd_mu])

    return sample_decay, sample_dropout, sample_wd
