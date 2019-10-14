import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from SegmentationUtils.progressBar import printProgressBar
import scipy.io as sio
import pdb

from torch import Tensor
from Typing import Set, Iterable, cast

import time


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


class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])

        return DiceN, DiceB , DiceW, DiceT


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)



def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(4))

    # ACDC
    Val[1] = 0.33333334
    Val[2] = 0.66666669
    Val[3] = 1.0

    x = predToSegmentation(pred)

    out = x * Val.view(1, 4, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotTumorClass(batch):
    data = batch.cpu().data.numpy()
    classLabels = np.zeros((data.shape[0], 2))

    tumorVal = 1.0
    for i in range(data.shape[0]):
        img = data[i, :, :, :]
        values = np.unique(img)
        if len(values) > 3:
            classLabels[i, 1] = 1
        else:
            classLabels[i, 0] = 1

    tensorClass = torch.from_numpy(classLabels).float()

    return Variable(tensorClass.cuda())


def getOneHotSegmentation(batch):
    backgroundVal = 0
    label1 = 0.33333334
    label2 = 0.66666669
    label3 = 1.0

    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3),
                             dim=1)
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334 # for ACDC this value
    #denom = 0.24705882 # for Chaos Dataset this value
    #denom = 0.25 # for Chaos Dataset this value
    #pdb.set_trace()
    #np.unique(batch.cpu().data.numpy())
    # temp = (batch / denom).round().long().squeeze()
    # temp = (batch / denom)
    # np.unique(temp.cpu().data.numpy())
    return (batch / denom).round().long().squeeze()

from scipy import ndimage



def saveImages(net, img_batch, batch_size, epoch, modelName):
    # print(" Saving images.....")
    path = 'SegmentationUtils/Results/Images/' + modelName

    if not os.path.exists(path):
        os.makedirs(path)
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)

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

    printProgressBar(total, total, done="Images saved !")


def inference(net, img_batch, batch_size, epoch):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)

    net.eval()

    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3 = dice(pred_y, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)

    return [ValDice1,ValDice2,ValDice3]



'''def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()




# TODO : use lr_scheduler from torch.optim
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


# TODO : use lr_scheduler from torch.optim
def adjust_learning_rate(lr_args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_args * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(" --- Learning rate:  {}".format(lr))'''
