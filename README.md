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
python src/main.py --task mnist --reg_coef1 0 --reg_coef2 0 --n_critic_iter 1 --search_space x
```