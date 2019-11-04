#!/usr/bin/env python3

import os
from functools import partial
from typing import Set, Iterable, cast

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor

tqdm_ = partial(tqdm, ncols=150,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


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


def saveImages(net, img_batch, batch_size, epoch, modelName):
    # print(" Saving images.....")
    path = 'Results/Images/' + modelName

    if not os.path.exists(path):
        os.makedirs(path)
    # total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()

    desc = f">> Validation ({epoch: 4d})"
    tq_iter = tqdm_(enumerate(img_batch), desc=desc)
    for i, data in tq_iter:
        # printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = image

        segmentation_prediction = net(MRI)
        pred_y = softMax(segmentation_prediction)

        segmentation = probs2class(pred_y)[:, None, ...].float()

        out = torch.cat((MRI, segmentation, labels))

        torchvision.utils.save_image(out.data, os.path.join(path,
                                                            f"{i}_Ep_{epoch:04d}.png"),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)
