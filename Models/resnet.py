import torch.nn as nn
import torch.nn.functional as F
import torch


class Resblock_2d(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, kernel_size):
        super(Resblock_2d, self).__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_dim, mid_dim, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(mid_dim, out_dim, kernel_size, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.layer(x) + x
