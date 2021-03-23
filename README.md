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
python3 src/gaussian_mixture.py  --n_epoch 601  --search_space full --n_critic_iter 2 --reg_coef1 0.1 --reg_coef2 1 --batch_size 64
```

MNIST

```bash
bash bin/mnist.sh

```

CIFAR10

```bash
bash bin/cifar10.sh
```
