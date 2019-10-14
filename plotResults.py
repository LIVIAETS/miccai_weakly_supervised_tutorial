import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt

# Tight bounds

# Loose bounds
# python plotResults.py ./Statistics/MIDL/FullySupervised ./Statistics/MIDL/WeaklySupervised_NO_SizeLoss ./Statistics/MIDL/Pathak_LooseBound ./Statistics/MIDL/WeaklySupervised_SizeLoss_LooseBound


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
    
def loadMetrics(folderName):

    CE_loss = np.load(folderName + '/CE_Losses.npy')
    sizeDiff = np.load(folderName + '/sizeDifferences.npy')

    
    return [CE_loss,sizeDiff]

def plot():

    model1Name = './Statistics/Weakly_Sup_CE_Loss'
    model2Name = './Statistics/Weakly_Sup_CE_Loss_SizePenalty'

    
    [CE_loss1, sizeDiff1] = loadMetrics(model1Name)
    [CE_loss2, sizeDiff2] = loadMetrics(model2Name)
    
    numEpochs1 = len(CE_loss1)
    numEpochs2 = len(CE_loss2)
    
    lim = numEpochs1
    if numEpochs2 < numEpochs1:
        lim = numEpochs2
        

    # Plot features
    xAxis = np.arange(0, lim, 1)
    
    plt.figure(1)

    # Training Dice
    plt.subplot(212)
    #plt.set_aspect('auto')
    plt.plot(xAxis, sizeDiff1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, sizeDiff2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    #plt.title('Size Difference')
    plt.ylabel('Size Difference')
    plt.grid(True)

    plt.subplot(211)
    plt.plot(xAxis, CE_loss1[0:lim], 'r-', label=model1Name)
    plt.plot(xAxis, CE_loss2[0:lim], 'b-', label=model2Name)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    #plt.title('CE Loss')
    plt.ylabel('CE Loss')
    plt.grid(True)

    plt.xlabel('Number of epochs')
    plt.show()

    
    
if __name__ == '__main__':
   plot()
