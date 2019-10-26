import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import pdb

######### ------ Cross-entropy + softmax loss function ---------- ###########
'''
   Cross-entropy loss (with softmax) for weakly supervised labels
   Unlike in the fully supervised case, pixels not included in the weak label
   are masked, and thus the cross-entropy is only computed for annotated pixels
   deriv of Softmax -> softmax - ground truth
'''
class CE_Loss_Weakly(torch.autograd.Function):

    def forward(self, input, target, weakLabels):
        self.save_for_backward(input, target, weakLabels)
        eps = 1e-10
        numPixelsNonMasked = weakLabels.sum()

        # -------  Stable softmax  -------
        input_numpy = input.cpu().numpy()
        exps = np.exp(input_numpy - np.max(input_numpy))
        sofMax = exps / np.sum(exps, axis=1)
        loss = - np.sum(np.log(sofMax[:,1,:,:])*(weakLabels.view(1,256,256)).cpu().numpy())/(numPixelsNonMasked.cpu().data.numpy()+eps)

        lossT =  torch.FloatTensor(1)
        lossT.fill_(np.float32(loss).item())

        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target,weakLabels = self.saved_variables
        numClasses = 2
        eps = 1e-10
        oneHotLabels = torch.cat((weakLabels == 0, weakLabels == 1), dim=0).view(1,numClasses,target.shape[1],target.shape[2]).float()

        # Mask the predictions to only annotated pixels
        mask=oneHotLabels
        mask[:,0,:,:]=0

        numPixelsNonMasked = weakLabels.sum()


        # Stable softmax
        input_numpy = input.cpu().data.numpy()
        exps = np.exp(input_numpy - np.max(input_numpy))
        softmax_y = exps / (np.sum(exps, axis=1))
        grad_input = (torch.Tensor(softmax_y).cuda() - torch.Tensor(oneHotLabels.cpu().data).cuda())*(torch.Tensor(mask.cpu().data).cuda())/(torch.Tensor(np.array(numPixelsNonMasked.cpu().data.numpy()+eps)).cuda())  # Divide by m or numPixelsNonMasked

        return grad_input.cuda(), None, None




######### ------ Size loss function  (naive way) ---------- ###########
# --- This function will push the prediction to be close ot sizeGT ---#
class Size_Loss_naive(torch.autograd.Function):
    ''' input is already the softmax now'''
    def forward(self, input, target):
        self.save_for_backward(input, target)

        softMax_np = input.cpu().numpy()
        softB = softMax_np[:,1,:,:]
        pixelsClassB = np.where( softB > 0.5 )  # TODO: Do threshold with pyTorch: http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html

        sizePred = len(pixelsClassB[0])

        # ---- Known sizes ------
        # Small circle = 7845
        sizeGT = 7845  # If we would have the target GT---> sizeGT = target.sum()

        loss = ((sizePred - sizeGT)**2)/(softB.shape[1]*softB.shape[2])

        lossT =  torch.FloatTensor(1)
        lossT.fill_(loss/100)  # 1000 is to weight this loss --> TODO: make this weight an input param
        lossT = lossT.cuda()

        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target = self.saved_variables

        m = input.shape[2]*input.shape[3]

        # Softmax  (It can be saved with save_for_backward??)
        softmax_y = input.cpu().data.numpy()
        softB = softmax_y[:,1,:,:]

        pixelsClassB = np.where( softB > 0.5 )
        sizePred = len(pixelsClassB[0])
        # Small circle = 7845
        sizeGT = 7845 # oneHotLabels[:,1,:,:].sum().cpu().data.numpy()

        grad_inputA = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0],1,softmax_y.shape[2],softmax_y.shape[3]),dtype='float32')

        grad_inputB.fill(2 * (sizePred-sizeGT)/(100*m))
        grad_input = np.concatenate((grad_inputA,grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None
