import numpy as np
import torch
from torch import nn
import ot
from matplotlib import pyplot as plt
import argparse
from pathlib import Path


class Generator2d(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_atoms, 2))

    def forward(self):
        return self.theta


def compute_wasserstein(target_sample, gen_sample, q, p):
    target_weights = np.ones(target_sample.shape[0]) / target_sample.shape[0]
    gen_weights = np.ones(gen_sample.shape[0]) / gen_sample.shape[0]
    M = torch.norm(target_sample[:, None, :] - gen_sample[None, :, :], dim=2, p=q)**p / p
    T = ot.emd(target_weights, gen_weights, M.detach().cpu().numpy())
    T = torch.FloatTensor(T)
    T.requires_grad_(False)
    W = ((M * T).sum())**(1./p)
    return W


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_atoms', type=int, default=10)
    parser.add_argument('--gen_atoms', type=int, default=7)
    parser.add_argument('-q', type=int, default=2)
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='figs')

    args= parser.parse_args()
    return args


def main(args):
    target_sample = torch.rand(args.target_atoms, 2) - .5
    
    generator = Generator2d(args.gen_atoms)
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)

    samples = []
    for it in range(args.n_iter):
        gen_sample = generator()
        #print(generator.theta, gen_sample)
        samples.append(np.array(gen_sample.detach().cpu().numpy()))
        W = compute_wasserstein(target_sample, gen_sample, args.q, args.p)
        optimizer.zero_grad()
        W.backward()
        optimizer.step()
    samples = np.stack(samples, 0)

    #gen_sample = generator().detach().cpu().numpy()
    target_sample = target_sample.detach().cpu().numpy()
    fig = plt.figure()
    for sample in samples[:-1]:
        plt.scatter(sample[:, 0], sample[:, 1], color='r', s=0.2)
    plt.scatter(target_sample[:, 0], target_sample[:, 1], color='b', s=100)
    plt.scatter(samples[-1, :, 0], samples[-1, :, 1], color='r')
    plt.title(f'p={args.p}')
    plt.plot([0], [0], marker='x', markersize=10, color='black')
    plt.savefig(Path(args.save_dir, f'discrete_p={args.p}.png'))
    plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

    