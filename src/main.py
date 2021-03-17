import torch
import numpy as np
import argparse
import tqdm
from torch import optim
from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import mnist
from qpwgan import QPWGAN

# fix for downloading MNIST
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

optim_params = {'lr': 1e-4, 'betas': (0.5, 0.999)}


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mnist',
            'cifar10',
            'gmm',
            'discrete'],
        default='mnist')
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('-q', '--q', type=int, default=2)
    parser.add_argument('-p', '--p', default=2)
    parser.add_argument('--n_critic_iter', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reg_coef1', type=float, default=0.1)
    parser.add_argument('--reg_coef2', type=float, default=0.1)

    args = parser.parse_args()
    return args


def main(args):
    if args.device is not None:
        device = torch.device(
            args.devce if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random_seed(args.seed)

    if args.task == 'mnist':
        # , transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]))
        train_dataset = datasets.MNIST('data', train=True, download=True)
        trainloader = DataLoader(train_dataset.data.reshape(-1, 28 * 28))
        generator = mnist.Generator().to(device)
        critic = mnist.Critic().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), **optim_params)
    critic_optimizer = optim.Adam(critic.parameters(), **optim_params)

    wgan = QPWGAN(generator,
                  critic,
                  trainloader,
                  gen_optimizer,
                  critic_optimizer,
                  q=args.q,
                  p=args.p,
                  n_epoch=args.n_epoch,
                  n_critic_iter=args.n_critic_iter,
                  verbose=True
                  )

    wgan.train()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
