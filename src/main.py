from functools import partial

import wandb
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

from models import mnist, cifar10
from qpwgan import QPWGAN
from utils import random_seed, DUMP_DIR, DATA_DIR

# fix for downloading MNIST
from six.moves import urllib

from utils import plotting_callback, save_callback, nearest_distance_callback, inception_callback, fidscore_callback

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

optim_params = {'lr': 1e-4, 'betas': (0.5, 0.999)}


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
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=int(5e4))
    parser.add_argument('-q', '--q', type=float, default=2)
    parser.add_argument('-p', '--p', type=float, default=2)
    parser.add_argument('--n_critic_iter', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-d', '--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reg_coef1', type=float, default=0.1)
    parser.add_argument('--reg_coef2', type=float, default=0.1)
    parser.add_argument('--search_space', type=str,
                        choices=['full', 'x'], default='x')
    parser.add_argument('--dump_dir', type=str, default=DUMP_DIR)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--wandb_entity', type=str, default='samokhinv')

    args = parser.parse_args()
    return args


def main(args):

    wandb.init(project='qp-wgan', entity=args.wandb_entity)
    wandb.config.update(args)
    wandb.run.name = f'{args.task}_q_{args.q}_p_{args.p}_critic_iter_{args.n_critic_iter}'

    if args.device is not None:
        device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random_seed(args.seed)

    Path(args.dump_dir).mkdir(exist_ok=True)

    norm_mean = 0.5  # 0.1307
    norm_std = 0.5  # 0.3081

    if args.task == 'mnist':
        transform = T.Compose(
            [T.ToTensor(),
             T.Normalize((norm_mean,), (norm_std,)),
             T.Lambda(lambda x: torch.flatten(x))
             ]
        )
        inv_normalize = T.Normalize(
            mean=(-norm_mean/norm_std,),
            std=(1/norm_std,)
        )

        train_dataset = datasets.MNIST(
            'data', train=True, download=True, transform=transform)
        trainloader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        generator = mnist.Generator().to(device)
        # generator.init_weights()
        critic = mnist.Critic().to(device)
        # critic.init_weights()

    elif args.task == 'cifar10':
        transform = T.Compose(
            [T.ToTensor(),
             T.Normalize(
                 (norm_mean,),
                 (norm_std,))
             ]
        )
        inv_normalize = T.Normalize(
            mean=(-norm_mean/norm_std,),
            std=(1/norm_std,)
        )
        train_dataset = datasets.CIFAR10(
            DATA_DIR, train=True, transform=transform, download=True)
        trainloader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        generator = cifar10.Generator().to(device)
        critic = cifar10.Critic().to(device)

    callbacks = list(
        map(
            lambda f: partial(f, dump_dir=args.dump_dir,
                              inv_normalize=inv_normalize),
            (plotting_callback, nearest_distance_callback, save_callback, inception_callback, fidscore_callback)
        )
    )

    gen_optimizer = optim.Adam(generator.parameters(), **optim_params)
    critic_optimizer = optim.Adam(critic.parameters(), **optim_params)

    n_epoch = int(args.n_iter / len(trainloader) / args.n_critic_iter)
#     n_epoch = args.n_epoch

    wandb.watch(generator)
    wandb.watch(critic)
    wgan = QPWGAN(generator,
                  critic,
                  trainloader,
                  gen_optimizer,
                  critic_optimizer,
                  q=args.q,
                  p=args.p,
                  n_epoch=n_epoch,
                  n_critic_iter=args.n_critic_iter,
                  search_space=args.search_space,
                  verbose=True,
                  reg_coef1=args.reg_coef1,
                  reg_coef2=args.reg_coef2,
                  task=args.task,  # for logging purposes
                  device=device
                  )

    wgan.train(callbacks=callbacks)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
