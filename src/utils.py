import torch
import torch.nn as nn
import numpy as np
#import random

DUMP_DIR = 'dump'
DATA_DIR = 'data'
FIGS_DIR = 'figs'
CHECKS_DIR = 'checkpoints'


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)
