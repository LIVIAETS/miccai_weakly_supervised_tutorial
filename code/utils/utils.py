#!/usr/bin/env python3

import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from skimage.io import imsave
from torch import Tensor, einsum

tqdm_ = partial(tqdm, ncols=125,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


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


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        imsave(str(save_path), seg.detach().cpu().numpy().astype(np.uint8))
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError("How did you get here")


# Save a fancy looking figure
def saveImages(net, img_batch, batch_size, epoch, dataset, mode, device):
    base_path = Path('results/') / dataset / mode
    (base_path / 'grids').mkdir(parents=True, exist_ok=True)  # combination of image, prediction and weak label
    (base_path / 'predictions').mkdir(parents=True, exist_ok=True)  # un-normalized prediction saved as png

    net.eval()

    desc = f">> Validation ({epoch: 4d})"

    log_dice = torch.zeros((len(img_batch)), device=device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        tq_iter = tqdm_(enumerate(img_batch), total=len(img_batch), desc=desc)
        for j, data in tq_iter:
            img = data["img"].to(device)
            weak_mask = data["weak_mask"].to(device)
            full_mask = data["full_mask"].to(device)

            logits = net(img)
            probs = F.softmax(5 * logits, dim=1)

            segmentation = probs2class(probs)[:, None, ...].float()
            log_dice[j] = dice_coef(probs2one_hot(probs), full_mask)[0, 1]  # 1st item, 2nd class

            out = torch.cat((img, segmentation, weak_mask[:, [1], ...]))

            torchvision.utils.save_image(out.data, base_path / 'grids' / f"{j}_Ep_{epoch:04d}.png",
                                         nrow=batch_size,
                                         padding=2,
                                         normalize=False,
                                         range=None,
                                         scale_each=False,
                                         pad_value=0)

            predicted_class: Tensor = probs2class(probs)
            filenames: List[str] = [Path(p).stem for p in data["path"]]

            save_images(predicted_class,
                        filenames,
                        base_path / 'predictions' / f"iter{epoch:03d}")

            tq_iter.set_postfix({"DSC": f"{log_dice[:j+1].mean():05.3f}"})
            tq_iter.update(1)
        tq_iter.close()


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


# Functions to supervise
def soft_size(a: Tensor) -> Tensor:
    return einsum("bkwh->bk", a)


def soft_centroid(a: Tensor) -> Tensor:
        b, k, w, h = a.shape

        # Generate all coordinates, on each axis
        grids = np.mgrid[0:w, 0:h]
        # for instance, this gives for a 256x256 img
        # produces two arrays of 256x256, containing the x and y coordinates for each pixel

        # tensor_grids = map_(lambda e: torch.tensor(e).to(a.device).type(torch.float32), grids)
        tensor_grids = [torch.tensor(g, dtype=torch.float32).to(a.device) for g in grids]

        # Make sure all grids have the same shape as img_shape
        assert all(tuple(t.shape) == (w, h)
                   for t in tensor_grids)

        flotted = a.type(torch.float32)
        tot = einsum("bk...->bk", flotted) + 1e-10
        assert tot.dtype == torch.float32

        centroids = [einsum("bkwh,wh->bk", flotted, grid) / tot
                     for grid in tensor_grids]
        assert all(e.dtype == torch.float32 for e in centroids), map_(lambda e: e.dtype, centroids)

        res = torch.stack(centroids, dim=2)
        assert res.shape == (b, k, 2)
        assert res.dtype == torch.float32

        return res
