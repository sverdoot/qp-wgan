import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class Generator(nn.Module):
    def __init__(self,
                 layers_dim: List[int] = [2, 128, 128, 2],
                 non_linear: nn.Module = nn.LeakyReLU(0.2),
                 output_non_linear: nn.Module = nn.LeakyReLU(0.2)):
        super(Generator, self).__init__()

        self.noise_dim = layers_dim[0]
        self.layers_dim = layers_dim
        layers_list = list()
        for i in range(len(layers_dim) - 1):
            layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            layers_list.append(
                output_non_linear if i == (
                    len(layers_dim) -
                    2) else non_linear)
        self.layer_seq = nn.Sequential(*layers_list)

    def forward(self, z):
        x = self.layer_seq(z)
        return x

    def sample(self, batch_size):
        z = (torch.rand(batch_size, self.noise_dim) - 0.5) * 2
        x = self.forward(z)
        return x


class Critic(nn.Module):
    def __init__(self,
                 layers_dim: List[int] = [2, 128, 128, 2],
                 non_linear: nn.Module = nn.LeakyReLU(0.2),
                 output_non_linear: nn.Module = nn.LeakyReLU(0.2)):
        super(Critic, self).__init__()

        self.input_dim = layers_dim[0]
        self.layers_dim = layers_dim
        layers_list = list()
        for i in range(len(layers_dim) - 1):
            layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            layers_list.append(
                output_non_linear if i == (
                    len(layers_dim) -
                    2) else non_linear)
        self.layer_seq = nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.layer_seq(x)
        return out
