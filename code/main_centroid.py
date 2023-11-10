#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path
from typing import Any, Callable, Tuple
from operator import itemgetter

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch import nn, einsum
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.dataset import (SliceDataset)
from utils.ShallowNet import (shallowCNN)
from utils.utils import (weights_init,
                         saveImages,
                         class2one_hot,
                         probs2one_hot,
                         tqdm_,
                         dice_coef,
                         soft_size,
                         soft_centroid)

from utils.losses import (ParametrableQuadraticPenalty,
                          ParametrableLogBarrier)


def setup(args) -> Tuple[nn.Module, Any, Any, DataLoader, DataLoader]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    num_classes = 2
    if args.dataset == 'TOY2':
        initial_kernels = 4
        print(">> Using a shallowCNN")
        net = shallowCNN(1, initial_kernels, num_classes)
        net.apply(weights_init)
    else:
        raise ValueError(args.dataset)
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    batch_size = 1
    root_dir = Path("data") / args.dataset

    transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    mask_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=2),
        itemgetter(0)
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             transform=transform,
                             mask_transform=mask_transform,
                             augment=True,
                             equalize=False)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           transform=transform,
                           mask_transform=mask_transform,
                           equalize=False)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            num_workers=5,
                            shuffle=False)

    return (net, optimizer, device, train_loader, val_loader)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader = setup(args)

    Loss_Size: Callable[[Tensor, Tensor], Tensor]
    Loss_Centroid: Callable[[Tensor, Tensor], Tensor]
    if args.mode == "quadratic":
        Loss_Size = ParametrableQuadraticPenalty(idk=[1], function=soft_size)
        Loss_Centroid = ParametrableQuadraticPenalty(idk=[1], function=soft_centroid)
    elif args.mode == "logbarrier":
        Loss_Size = ParametrableLogBarrier(idk=[1], function=soft_size)
        Loss_Centroid = ParametrableLogBarrier(idk=[1], function=soft_centroid)

    for i in range(args.epochs):
        net.train()

        log_dice = torch.zeros((len(train_loader)), device=device)

        desc = f">> Training   ({i: 4d})"
        tq_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=desc)
        for j, data in tq_iter:
            img = data["img"].to(device)
            full_mask = data["full_mask"].to(device)

            # Sanity tests to see we loaded and encoded the data correctly
            assert 0 <= img.min() and img.max() <= 1
            B, _, W, H = img.shape
            _, K, _, _ = full_mask.shape
            assert B == 1  # Since we log the values in a simple way, doesn't handle more

            true_size: Tensor = soft_size(full_mask)[..., None]  # Add an extra axis
            assert true_size.shape == (B, K, 1)  # last one is dimensionality of the value computed (size)
            true_centroid: Tensor = soft_centroid(full_mask)
            assert true_centroid.shape == (B, K, 2)  # Dimensionality is two for the centroid (two axis)

            bounds_size = einsum("bkm,u->bkmu", true_size, torch.tensor([0.9, 1.1],
                                                                        dtype=torch.float32,
                                                                        device=true_size.device))
            bounds_centroid = einsum("bkm,u->bkmu", true_centroid, torch.tensor([0.9, 1.1],
                                                                                dtype=torch.float32,
                                                                                device=true_size.device))

            optimizer.zero_grad()

            logits = net(img)
            pred_softmax = F.softmax(5 * logits, dim=1)
            pred_seg = probs2one_hot(pred_softmax)

            log_dice[j] = dice_coef(pred_seg, full_mask)[0, 1]  # 1st item, 2nd class

            combined_loss = Loss_Size(pred_softmax, bounds_size) + Loss_Centroid(pred_softmax, bounds_centroid)

            combined_loss.backward()
            optimizer.step()

            tq_iter.set_postfix({"DSC": f"{log_dice[:j+1].mean():05.3f}"})
            tq_iter.update(1)
        tq_iter.close()

        if args.mode == "logbarrier":
            Loss_Size.t *= 1.1
            Loss_Centroid.t *= 1.1

        if (i % 5) == 0:
            saveImages(net, val_loader, 1, i, args.dataset, args.mode, device)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=['TOY2'])
    parser.add_argument('--mode', default='quadratic', choices=['quadratic', 'logbarrier'])

    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    runTraining(args)


if __name__ == '__main__':
    main()
