# qp-wgan

```bash
pip install -r requirements.txt
pip install -e .
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