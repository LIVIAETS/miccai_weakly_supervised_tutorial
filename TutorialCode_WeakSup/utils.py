#!/usr/bin/env python3

import os
from typing import Set, Iterable, cast

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = torch.zeros(4)

    # ACDC
    Val[1] = 0.33333334
    Val[2] = 0.66666669
    Val[3] = 1.0

    x = predToSegmentation(pred)
    _, _1, *img_shape = x.shape

    padded = torch.zeros(1, 4, *img_shape)
    padded[:, 0, ...] = x[:, 0, ...]
    padded[:, 3, ...] = x[:, 1, ...]

    out = padded * Val[None, :, None, None]
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation(batch):
    backgroundVal = 0
    label1 = 0.33333334
    label2 = 0.66666669
    label3 = 1.0

    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3),
                             dim=1)
    return oneHotLabels.float()


def saveImages(net, img_batch, batch_size, epoch, modelName):
    # print(" Saving images.....")
    path = 'Results/Images/' + modelName

    if not os.path.exists(path):
        os.makedirs(path)
    # total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()

    for i, data in tqdm(enumerate(img_batch)):
        # printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = image
        Segmentation = labels

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)

        segmentation = getSingleImage(pred_y)

        out = torch.cat((MRI, segmentation, Segmentation))

        torchvision.utils.save_image(out.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)

    print("Images saved !")

    # printProgressBar(total, total, done="Images saved !")
