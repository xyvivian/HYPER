import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, size, m=1, n=10000):
        super().__init__()
        assert size % 2 == 0
        self.size = size
        self.m = m
        self.n = n

    def forward(self, value):
        index = torch.arange(self.size // 2)
        denominator = torch.pow(self.n, 2 * index / self.size)
        out1 = torch.sin(self.m * value / denominator)
        out2 = torch.cos(self.m * value / denominator)
        return torch.stack([out1, out2], dim=1).view(-1)

    def arch_pos_emb(self, value):
        pos_lst = []
        for i in value:
            pos_lst.append(self.forward(torch.tensor(i)))
        return torch.stack(pos_lst, dim=0).view(-1)
