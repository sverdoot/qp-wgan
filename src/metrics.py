import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from typing import Tuple, List

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from scipy.stats import entropy


def inception_score(imgs: torch.Tensor,
                    device: torch.device = torch.device('cuda'),
                    batch_size: int = 64,
                    resize=False,
                    splits=1
                    ) -> Tuple[float, float]:
    '''
    from https://github.com/sbarratt/inception-score-pytorch 
    '''

    """Computes the inception score of the generated images imgs
    imgs -- Torch Tensor of (Nx3xHxW) numpy images normalized in the range [-1, 1]
    device -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]

    assert batch_size > 0
    assert N > batch_size

    if device == torch.device('cuda'):
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    dataset = torch.utils.data.TensorDataset(imgs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    inception_model = inception_v3(
        pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)

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
    
    del inception_model
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


def closest_samples(dataset: torch.Tensor, generated: torch.Tensor) -> List[float]:
    distances = torch.cdist(dataset, generated)
    return distances.min(0).values.tolist()


def get_activations(model, images: torch.Tensor, batch_size: int = 200, dims: int = 2048, device: torch.device = torch.device('cuda')):
    '''
    from https://github.com/mseitzer/pytorch-fid
    '''
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- model       : Instance of inception model
    -- images       : torch Tensor of images
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for batch in dataloader:
        
        batch = batch[0].to(device)
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def compute_statistics_of_image(model, images: torch.Tensor, batch_size: int = 200, dims: int = 2048,
                                device: torch.device = torch.device('cuda')):
    act = get_activations(model, images, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_score(images_a: torch.Tensor, images_b: torch.Tensor, batch_size: int = 64, device: torch.device = torch.device('cuda'), dims: int = 2048) -> float:
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_image(
        model, images_a, batch_size, dims, device)
    m2, s2 = compute_statistics_of_image(
        model, images_b, batch_size, dims, device)
    
    del model

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
