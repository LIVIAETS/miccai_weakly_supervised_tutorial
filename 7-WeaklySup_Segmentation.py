#!/usr/bin/env python3

import os
import argparse
from operator import add
from functools import reduce

import torch
import numpy as np
from torch import nn
from torch import einsum
from torch.utils.data import DataLoader
from torchvision import transforms

from TutorialCode_WeakSup.medicalDataLoader import (MedicalImageDataset)
from TutorialCode_WeakSup.ShallowNet import (shallowCNN)
from TutorialCode_WeakSup.utils import (weights_init,
                                        saveImages,
                                        probs2one_hot,
                                        sset,
                                        tqdm_)

from TutorialCode_WeakSup.losses import (Size_Loss_naive,
                                         CE_Loss_Weakly)


def runTraining(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('  ## Mode loss: {} ##'.format(args.mode))
    print('-' * 40)

    batch_size = 1
    batch_size_val_savePng = 1
    lr = 0.0005
    epoch = args.epochs
    circle_size = 7845
    mode = args.mode  # 0-> Only CE   1 -> CE + Size loss
    root_dir = 'TutorialCode_WeakSup/Data/ToyExample'

    if mode == 0:
        modelName = 'Weakly_Sup_CE_Loss'
    else:
        modelName = 'Weakly_Sup_CE_Loss_SizePenalty'

    print(f' {root_dir} ')
    transform = transforms.Compose([
        transforms.ToTensor()  # Keep in mind, divide by 255 and do weird things
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()  # Keep in mind, divide by 255 and do weird things
    ])

    train_set = MedicalImageDataset('train',
                                    root_dir,
                                    transform=transform,
                                    mask_transform=mask_transform,
                                    augment=True,
                                    equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

    val_set = MedicalImageDataset('val',
                                  root_dir,
                                  transform=transform,
                                  mask_transform=mask_transform,
                                  equalize=False)

    val_loader_save_imagesPng = DataLoader(val_set,
                                           batch_size=batch_size_val_savePng,
                                           num_workers=5,
                                           shuffle=False)

    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = 2
    initial_kernels = 4

    # myNetwork
    net = shallowCNN(1, initial_kernels, num_classes)

    # Initialize
    net.apply(weights_init)

    # Define losses and softmax
    softMax = nn.Softmax()
    cross_entropy_loss_weakly = CE_Loss_Weakly()
    sizeLoss = Size_Loss_naive()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    Losses_CE = []
    Losses_Size = []
    sizeDifferences = []
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()

        lossValCE = []
        lossValSize = []
        sizeDiff = []

        desc = f">> Training   ({i})"
        tq_iter = tqdm_(enumerate(train_loader), desc=desc)
        for j, data in tq_iter:
            image, labels, img_names = data

            optimizer.zero_grad()
            MRI = image
            Segmentation = labels.long()
            assert 0 <= MRI.min() and MRI.max() <= 1
            assert sset(Segmentation, [0, 1])

            segmentation_prediction = net(MRI)
            segment_prob = softMax(segmentation_prediction)

            pred_circle_size = einsum("bcwh->bc", probs2one_hot(segment_prob))[:, 1]
            sizeDiff.append((pred_circle_size - circle_size).abs().float().mean())

            lossCE = cross_entropy_loss_weakly(segment_prob, Segmentation[:, 0, ...])
            assert lossCE.requires_grad

            if mode == 0:
                lossEpoch = lossCE
                lossValSize.append(0)
            else:
                sizeLoss_val = sizeLoss(segment_prob)
                assert sizeLoss_val.requires_grad
                lossEpoch = lossCE + sizeLoss_val
                # lossEpoch = reduce(add, [lossCE, sizeLoss_val])
                assert lossEpoch.requires_grad

                lossValSize.append(sizeLoss_val.item())

            net.zero_grad()
            lossEpoch.backward()
            optimizer.step()

            lossValCE.append(lossCE.item())

            tq_iter.set_postfix({"SizeDiff": f"{np.mean(sizeDiff):6.1f}",
                                 "LossCE": f"{np.mean(lossValCE):5.2e}",
                                 **({"LossSize": f"{np.mean(lossValSize):5.2e}"} if mode == 1 else {})})

        sizeDifferences.append(np.mean(sizeDiff))
        Losses_CE.append(np.mean(lossValCE))
        Losses_Size.append(np.mean(lossValSize))

        directory = 'Results/Statistics/' + modelName
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'CE_Losses.npy'), Losses_CE)
        np.save(os.path.join(directory, 'Losses_Size.npy'), Losses_Size)
        np.save(os.path.join(directory, 'sizeDifferences.npy'), sizeDifferences)

        if (i % 10) == 0:
            saveImages(net, val_loader_save_imagesPng, batch_size_val_savePng, i, modelName)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--mode', default=0, type=int)
    args = parser.parse_args()
    runTraining(args)


if __name__ == '__main__':
    main()
