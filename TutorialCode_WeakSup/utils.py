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




############### To Save gradients ##############
def extract(xVar):
    global yGrad
    yGrad = xVar

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    #print('')
    #print('input: ', type(input))
    #print('input[0]: ', type(input[0]))
    #print('output: ', type(output))
    #print('')
    #print('input size:', input[0].size())
    #print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    #print('Inside class:' + self.__class__.__name__)
    #print('')
    #print('grad_input: ', type(grad_input))
    #print('grad_input[0]: ', type(grad_input[0]))
    #print('grad_output: ', type(grad_output))
    #print('grad_output[0]: ', type(grad_output[0]))
    #print('')
    #print('grad_input size:', grad_input[0].size())
    #print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

def printgradLoss(self, grad_input, grad_output):
    print('-->Grad through the loss '  + self.__class__.__name__ + ' : {}'.format(grad_input))
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input_Norm:', grad_input[0].data.norm())
    pdb.set_trace()
    

#########################################
    
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        



def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    numClasses = 2
    Val = to_var(torch.zeros(numClasses))
    #Val[1] = 0.33333334
    #Val[2] = 0.66666669
    Val[1] = 1.0
    
    x = predToSegmentation(pred)
    
    out = x * Val.view(1, 2, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    spineLabel = 0.33333334
    return (batch / spineLabel).round().long().squeeze()

    
    
def saveImages(net, img_batch, batch_size, epoch, modelName, deepSupervision = False):
    # print(" Saving images.....")
    path = 'TutorialCode_WeakSup/Results/Images/' + modelName

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
            
        if deepSupervision == False:
            # No deep supervision
            segmentation_prediction = net(MRI)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        segment_prob = softMax(segmentation_prediction)
        segmentation = getSingleImage(segment_prob)

        segment_circle = (segment_prob[:,1,:,:] > 0.5 ).view((segment_prob.shape[2], segment_prob.shape[3])).cpu().data.numpy() 
        diffSize = abs(segment_circle.sum()-7845)
           
        print('(VAL) Size difference: {}'.format(diffSize))    
        out = torch.cat((MRI, segmentation, Segmentation))
   
        torchvision.utils.save_image(out.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)
    printProgressBar(total, total, done="Images saved !")

  
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

