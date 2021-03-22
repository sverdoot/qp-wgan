# qp-wgan

```bash
pip install -r requirements.txt
pip install -e .
```

To use wandb tracking, do in advance
```bash
wandb login
```

Optimization of Wasserstein metric on discrete measure

```bash
python src/discrete_measures.py
```

Approximating a Gaussian mixture distribution

```bash
python src/gaussian_mixture.py --n_epoch 601 --search_space full
```

MNIST

```bash
bash bin/mnist.sh

```

CIFAR10

```bash
bash bin/cifar10.sh
```

CIFAR10 Progress

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="480px" src="figs/gif/cifar_1_1_1.gif"></td>
    <td><img width="480px" src="figs/gif/cifar_1_1_5.gif"></td>
  </tr>
  <tr>
    <td>q=1, p=1, critic iters = 1</td>
    <td>q=1, p=1, critic iters = 5</td>
  </tr>
    <tr>
    <td><img width="480px" src="figs/gif/cifar_1_2_1.gif"></td>
    <td><img width="480px" src="figs/gif/cifar_1_2_5.gif"></td>
  </tr>
  <tr>
    <td>q=1, p=2, critic iters = 1</td>
    <td>q=1, p=2, critic iters = 5</td>
  </tr>
    <tr>
    <td><img width="480px" src="figs/gif/cifar_2_2_1.gif"></td>
    <td><img width="480px" src="figs/gif/cifar_2_2_5.gif"></td>
  </tr>
  <tr>
    <td>q=2, p=2, critic iters = 1</td>
    <td>q=2, p=2, critic iters = 5</td>
  </tr>

</table>
