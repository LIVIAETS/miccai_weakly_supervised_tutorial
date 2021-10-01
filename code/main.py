#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Tuple
from operator import itemgetter

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.dataset import (SliceDataset)
from utils.ShallowNet import (shallowCNN)
from utils.residual_unet import ResidualUNet
from utils.utils import (weights_init,
                         saveImages,
                         class2one_hot,
                         probs2one_hot,
                         one_hot,
                         tqdm_,
                         dice_coef)

from utils.losses import (CrossEntropy,
                          PartialCrossEntropy,
                          NaiveSizeLoss)


def setup(args) -> Tuple[nn.Module, Any, Any, DataLoader, DataLoader]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    num_classes = 2
    if args.dataset == 'TOY':
        initial_kernels = 4
        print(">> Using a shallowCNN")
        net = shallowCNN(1, initial_kernels, num_classes)
        net.apply(weights_init)
    else:
        print(">> Using a fully residual UNet")
        net = ResidualUNet(1, num_classes)
        net.init_weights()
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

    ce_loss = CrossEntropy(idk=[0, 1])  # Supervise both background and foreground
    partial_ce = PartialCrossEntropy()  # Supervise only foregroundz
    sizeLoss = NaiveSizeLoss()

    for i in range(args.epochs):
        net.train()

        log_ce = torch.zeros((len(train_loader)), device=device)
        log_sizeloss = torch.zeros((len(train_loader)), device=device)
        log_sizediff = torch.zeros((len(train_loader)), device=device)
        log_dice = torch.zeros((len(train_loader)), device=device)

        desc = f">> Training   ({i: 4d})"
        tq_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=desc)
        for j, data in tq_iter:
            img = data["img"].to(device)
            full_mask = data["full_mask"].to(device)
            weak_mask = data["weak_mask"].to(device)

            bounds = data["bounds"].to(device)

            optimizer.zero_grad()

            # Sanity tests to see we loaded and encoded the data correctly
            assert 0 <= img.min() and img.max() <= 1
            B, _, W, H = img.shape
            assert B == 1  # Since we log the values in a simple way, doesn't handle more
            assert weak_mask.shape == (B, 2, W, H)
            assert one_hot(weak_mask), one_hot(weak_mask)

            logits = net(img)
            pred_softmax = F.softmax(5 * logits, dim=1)
            pred_seg = probs2one_hot(pred_softmax)

            pred_size = einsum("bkwh->bk", pred_seg)[:, 1]
            log_sizediff[j] = pred_size - data["true_size"][0, 1]
            log_dice[j] = dice_coef(pred_seg, full_mask)[0, 1]  # 1st item, 2nd class

            if args.mode == 'full':
                ce_val = ce_loss(pred_softmax, full_mask)
                log_ce[j] = ce_val.item()

                log_sizeloss[j] = 0

                lossEpoch = ce_val
            elif args.mode == 'unconstrained':
                ce_val = partial_ce(pred_softmax, weak_mask)
                log_ce[j] = ce_val.item()

                log_sizeloss[j] = 0

                lossEpoch = ce_val
            else:
                ce_val = partial_ce(pred_softmax, weak_mask)
                log_ce[j] = ce_val.item()

                sizeLoss_val = sizeLoss(pred_softmax, bounds)
                log_sizeloss[j] = sizeLoss_val.item()

                lossEpoch = ce_val + sizeLoss_val

            lossEpoch.backward()
            optimizer.step()

            tq_iter.set_postfix({"DSC": f"{log_dice[:j+1].mean():05.3f}",
                                 "SizeDiff": f"{log_sizediff[:j+1].mean():07.1f}",
                                 "LossCE": f"{log_ce[:j+1].mean():5.2e}",
                                 **({"LossSize": f"{log_sizeloss[:j+1].mean():5.2e}"} if args.mode == 'constrained' else {})})
            tq_iter.update(1)
        tq_iter.close()

        if (i % 5) == 0:
            saveImages(net, val_loader, 1, i, args.dataset, args.mode, device)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY', choices=['TOY', 'PROMISE12'])
    parser.add_argument('--mode', default='unconstrained', choices=['constrained', 'unconstrained', 'full'])

    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    runTraining(args)


if __name__ == '__main__':
    main()
