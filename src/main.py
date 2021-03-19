import torch
import numpy as np
import argparse
import tqdm
from torch import optim
from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T
from matplotlib import pyplot as plt

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
    parser.add_argument('-p', '--p', type=int, default=2)
    parser.add_argument('--n_critic_iter', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reg_coef1', type=float, default=0.1)
    parser.add_argument('--reg_coef2', type=float, default=0.1)
    parser.add_argument('--search_space', type=str, choices=['full', 'x'], default='x')

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
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,)), T.Lambda(lambda x: torch.flatten(x))])
        inv_normalize = T.Normalize(
            mean=[-0.1307/0.3081,],
            std=[1/0.3081,]
        )

        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        generator = mnist.Generator().to(device)
        #generator.init_weights()
        critic = mnist.Critic().to(device)
        #critic.init_weights()

        def callback(wgan, epoch, *args, **kwargs):
            if epoch % 1 != 0:
                return
            sample = wgan.generator.sample(100, device=wgan.device)
            sample = sample.reshape(-1, 28, 28)
            sample = inv_normalize(sample).detach().cpu().numpy()
            _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
            #plt.title(f'({wgan.q}, {wgan.p})-WGAN')
            for ax, im in zip(axs.flat, sample):
                ax.imshow(im)
                ax.set_aspect('equal')
                ax.axis('off')
            plt.savefig(Path('../test', f'{wgan.q}_{wgan.p}_mnist_{epoch}epoch.pdf'))
            plt.close()

            data = train_dataset.data[:100]
            _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
            for ax, im in zip(axs.flat, data):
                ax.imshow(im)
                ax.set_aspect('equal')
                ax.axis('off')
            plt.savefig(Path('../test', f'mnist.pdf'))
            plt.close()

        callbacks = callback

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
                  search_space=args.search_space,
                  verbose=True,
                  reg_coef1=args.reg_coef1,
                  reg_coef2=args.reg_coef2,
                  )

    wgan.train(callbacks=callbacks)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
