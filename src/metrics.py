# TODO: Inception score, Fresche score for Cifar10, distance to nearest for MNIST

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs: torch.Tensor, 
                    device: torch.device = torch.device('cuda'), 
                    batch_size: int = 128, 
                    resize=False, 
                    splits=1
                   ):
    """Computes the inception score of the generated images imgs
    imgs -- Torch Tensor of (Nx3xHxW) numpy images normalized in the range [-1, 1]
    device -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]
#     N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    if device == torch.device('cuda'):
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    dataset = torch.utils.data.TensorDataset(imgs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)