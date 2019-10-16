#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def loadMetrics(folderName):
    CE_loss = np.load(folderName + '/CE_Losses.npy')
    sizeDiff = np.load(folderName + '/sizeDifferences.npy')
    size_loss = np.load(folderName + '/Losses_Size.npy')

    return [CE_loss, sizeDiff, size_loss]


def plot():
    model1Name = 'Results/Statistics/Weakly_Sup_CE_Loss'
    model2Name = 'Results/Statistics/Weakly_Sup_CE_Loss_SizePenalty'

    [CE_loss1, sizeDiff1, size_loss_1] = loadMetrics(model1Name)
    [CE_loss2, sizeDiff2, size_loss_2] = loadMetrics(model2Name)

    numEpochs1 = len(CE_loss1)
    numEpochs2 = len(CE_loss2)

    lim = min(numEpochs1, numEpochs2)

    # Plot features
    xAxis = np.arange(0, lim, 1)

    plt.figure(1)

    # Training Dice
    plt.subplot(311)
    plt.plot(xAxis, CE_loss1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, CE_loss2[0:lim], 'b-', label=model2Name)
    plt.legend(loc='upper center', shadow=True, fontsize='large')
    plt.ylabel('CE Loss')
    plt.grid(True)

    plt.subplot(312)
    plt.plot(xAxis, sizeDiff1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, sizeDiff2[0:lim], 'b-', label=model2Name)
    plt.legend(loc='upper center', shadow=True, fontsize='large')
    plt.ylabel('Size Difference')
    plt.grid(True)

    plt.subplot(313)
    plt.plot(xAxis, size_loss_1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, size_loss_2[0:lim], 'b-', label=model2Name)
    plt.legend(loc='upper center', shadow=True, fontsize='large')
    plt.ylabel('Size Loss')
    plt.grid(True)

    plt.xlabel('Number of epochs')
    plt.show()


if __name__ == '__main__':
    plot()
