#!/usr/bin/env python3

import os
import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from SegmentationUtils.progressBar import printProgressBar
from TutorialCode_WeakSup.medicalDataLoader import (MedicalImageDataset)
from TutorialCode_WeakSup.ShallowNet import (shallowCNN)
from TutorialCode_WeakSup.utils import (weights_init,
                                        saveImages)

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
    img_size = 256
    mode = args.mode  # 0-> Only CE   1 -> CE + Size loss
    root_dir = 'TutorialCode_WeakSup/Data/ToyExample'

    print(' {} '.format(root_dir))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
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

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    Losses_CE = []
    Losses_Size = []
    sizeDifferences = []
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):

        net.train()

        totalImages = len(train_loader)
        lossValCE = []
        lossValSize = []
        sizeDiff = []

        for j, data in enumerate(train_loader):
            image, labels, img_names = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            MRI = image
            Segmentation = labels.long()

            # ################## Train  ###################

            segmentation_prediction = net(MRI)
            segment_prob = softMax(segmentation_prediction)

            # # ------ Compute some metrics -------- ##
            segment_circle = (segment_prob[:, 1, :, :] > 0.5).view((segment_prob.shape[2], segment_prob.shape[3])).cpu().data.numpy()
            sizeDiff.append(abs(segment_circle.sum() - circle_size))

            # ##   ------ Define the losses --- ####
            # ---- CE weakly loss ------ #s
            # It will get ideally prediction, GT and weak labels (to mask the pixels non annotated)
            lossCE = cross_entropy_loss_weakly(segment_prob, Segmentation[:, 0, ...])

            # ----- Size losses ------ #
            sizeLoss_val = sizeLoss(segment_prob)

            if mode == 0:
                lossEpoch = lossCE
            else:
                lossEpoch = lossCE + sizeLoss_val
            # lossEpoch = sizeLoss_val
            net.zero_grad()
            lossEpoch.backward(retain_graph=True)

            optimizer.step()

            lossValCE.append(lossCE.item())
            lossValSize.append(sizeLoss_val.item())

        sizeDifferences.append(np.mean(sizeDiff))
        Losses_CE.append(np.mean(lossValCE))
        Losses_Size.append(np.mean(lossValSize))

        printProgressBar(totalImages, totalImages,
                         done=f"[Training] Epoch: {i}, LossCE: {np.mean(lossValCE):.4f}, LossSize: {np.mean(lossValSize):.4f}, SizeDiff: {np.mean(sizeDiff)} ")

        if mode == 0:
            modelName = 'Weakly_Sup_CE_Loss'
        else:
            modelName = 'Weakly_Sup_CE_Loss_SizePenalty'

        directory = 'TutorialCode_WeakSup/Results/Statistics/' + modelName
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'CE_Losses.npy'), Losses_CE)
        np.save(os.path.join(directory, 'Losses_Size.npy'), Losses_Size)
        np.save(os.path.join(directory, 'sizeDifferences.npy'), sizeDifferences)

        if (i % 10) == 0:
            saveImages(net, val_loader_save_imagesPng, batch_size_val_savePng, i, modelName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--mode', default=0, type=int)
    args = parser.parse_args()
    runTraining(args)
