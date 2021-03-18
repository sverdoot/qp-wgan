import numpy as np
import torch
from torch import nn
import ot
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from typing import List, Union

from models.gaussian_model import Generator, Critic
from qpwgan import QPWGAN
from main import optim_params, random_seed

# optim_params = {'lr': 1e-4, 'betas': (0.5, 0.999)}

def generate_2d_gmm(n_data: Union[int, List[int]],
                    mu_vector: np.array, variance_vector: np.array) -> np.array:
    assert mu_vector.ndim == 2
    assert mu_vector.shape[1] == 2
    assert variance_vector.ndim == 1 or variance_vector.ndim == 3
    if not isinstance(n_data, List):
        n_data = [n_data // mu_vector.shape[0]] * mu_vector.shape[0]
    if variance_vector.ndim == 1:
        i_matrix = np.eye(2)
        return np.concatenate(
            [np.random.multivariate_normal(mu_vector[i_id, :], i_matrix * variance_vector, i, check_valid='ignore') for i_id,
             i in enumerate(n_data)]).astype(np.float32)
    else:
        return np.concatenate(
            [np.random.multivariate_normal(mu_vector[i_id, :], variance_vector[i_id], i, check_valid='ignore') for i_id, i in
             enumerate(n_data)]).astype(np.float32)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=int, default=2)
    parser.add_argument('-p', nargs='+', type=int, default=[1, 2, 5])
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--n_critic_iter', type=int, default=5)
    parser.add_argument('--number_of_clusters', type=int, default=None)
    parser.add_argument('--amount_of_points', type=int, default=None)
    parser.add_argument(
        '--points_per_cluster',
        nargs='+',
        default=[
            60,
            30,
            50])
    parser.add_argument('--save_dir', type=str, default='figs')
    parser.add_argument('--search_space', type=str, choices=['full', 'x'], default='x')
    parser.add_argument('--reg_coef1', type=float, default=1.)
    parser.add_argument('--reg_coef2', type=float, default=1.)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.amount_of_points is not None:
        args.points_per_cluster = [
            args.amount_of_points //
            args.number_of_clusters for _ in range(
                args.number_of_clusters)]
    else:
        args.number_of_clusters = len(args.points_per_cluster)

#     mus = np.random.randn(2,args.number_of_clusters).T * 2
#     mus = ((np.random.random(size=(2, args.number_of_clusters)) - 0.5) * 2).T
    mus = np.array([[-0.7, -0.3], [0.4, 0.8], [0.8, -0.5]])
    variances = np.abs(np.random.randn(args.number_of_clusters, 2, 2)) / 100
    target_sample = generate_2d_gmm(n_data=args.points_per_cluster,
                                    mu_vector=mus,
                                    variance_vector=variances
                                    )

    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()

    for i, p in enumerate(args.p):
        print("=" * 50)
        print(f"q {args.q} - p {p} metric")

        generator = Generator().to(device)
        #generator.init_weights()
        critic = Critic().to(device)
        #critic.init_weights()
        gen_optimizer = torch.optim.Adam(
            generator.parameters(), **optim_params)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), **optim_params)

        wgan = QPWGAN(generator,
                      critic,
                      None,
                      gen_optimizer,
                      critic_optimizer,
                      p=p,
                      device=device,
                      verbose=True,
                      n_critic_iter=args.n_critic_iter,
                      search_space=args.search_space,
                      reg_coef1=args.reg_coef1,
                      reg_coef2=args.reg_coef2,
                      )
        gen_loss_history, wass_history = wgan.train_gaussian_mixture(torch.Tensor(target_sample), args.n_iter)
        sample = wgan.generator.sample(
            batch_size=100, device=device).to('cpu').detach().numpy()

        fig = plt.figure()
        plt.scatter(sample[:, 0], sample[:, 1],
                    color='red', s=10, label='generated')
        plt.scatter(target_sample[:, 0], target_sample[:, 1],
                    color='cornflowerblue', s=10, label='original data')
        plt.title(f'p={p}')
        plt.plot([0], [0], marker='x', markersize=10, color='black')
        plt.legend()
        plt.savefig(Path(args.save_dir, f'sampled_gaussian_p={p}.pdf'))
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(gen_loss_history)), gen_loss_history, c='b')
        plt.plot(np.arange(len(wass_history)), wass_history, c='r')
        plt.title(f'{p}-WGAN')
        plt.xlabel(r'$N_{epoch}$')
        plt.grid()
        plt.savefig(Path(args.save_dir, f'gaussian_loss_p={p}.pdf'))
        plt.close()

#         ax.plot(np.arange(args.n_iter), wass, label=f'p={p}', color=colors[i])
#         ax2.plot(np.arange(args.n_iter), grad_norms, linestyle='--', color=colors[i])
    #     ax.grid()
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel(r'p-Wasserstein distance')

#     ax2.yaxis.set_label_position("right")
#     ax2.set_ylabel(r'$\|\nabla_{\theta} W_p(g_{\theta}, \mu)\|_2$')

#     ax.legend()
    # ax2.legend()
    # plt.savefig(Path(args.save_dir, f'sampled_gaussian_p={args.p}.png'))
    # plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
