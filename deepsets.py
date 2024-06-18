import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class DeepSet(nn.Module):
    def __init__(self, input_dim=1, feature_dim=10, multiplier=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.phi = nn.Linear(input_dim, feature_dim)
        self.rho1 = nn.Linear(feature_dim * 1, feature_dim * multiplier)
        self.rho2 = nn.Linear(feature_dim * multiplier, 1)
        self.reset_parameters()

    def forward(self, x):
        x = self.phi(x)
        x_max, _ = torch.max(x, dim=0)
        x_rep = x_max
        x = self.rho1(x_rep)
        x = F.relu(x)
        x = self.rho2(x)
        # x = F.relu(x)
        return x, x_rep

    def reset_parameters(self):
        """ Initialize the weights and bias.
        :return: None
        """
        torch.nn.init.xavier_uniform_(self.phi.weight)
        torch.nn.init.xavier_uniform_(self.rho1.weight)
        torch.nn.init.xavier_uniform_(self.rho2.weight)
