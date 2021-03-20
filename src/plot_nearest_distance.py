import json
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_with_files', type=str, default='dump')
    parser.add_argument('--folder_folder_to_save', type=str, default='figs')
    args = parser.parse_args()
    return args


def main(args):

    plt.figure(figsize=(14, 10))
    for i in Path(args.folder_with_files).iterdir():
        if str(i).endswith('39.json'):  # hardcoded
            data = json.load(i.open('r'))
            q, p, critic = str(i).split('__')[1:-1]
            q = float(q)
            p = float(p)
            critic = int(critic.split('_')[1])

            if critic == 5:
                if p != 1.2:
                    label = f'({int(q)}-{int(p)})'
                else:
                    label = f'({q}-{p}) critic iter {critic}'
                sns.kdeplot(data=data, label=label)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Distance to Nearest', fontsize=35)
    plt.ylabel('Frequency', fontsize=35)
    plt.legend(fontsize=22)
#     plt.show()
    plt.savefig(Path(args.folder_folder_to_save, 'nearest_distance_mnist.pdf'))
    plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
