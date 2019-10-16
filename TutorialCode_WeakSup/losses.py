#!/usr/bin/env python3

import torch
from torch import einsum

# from TutorialCode_WeakSup.utils import class2one_hot
from .utils import class2one_hot, simplex


# ######## ------ Cross-entropy + softmax loss function ---------- ###########
class CE_Loss_Weakly():
    '''
       Cross-entropy loss (with softmax) for weakly supervised labels
       Unlike in the fully supervised case, pixels not included in the weak label
       are masked, and thus the cross-entropy is only computed for annotated pixels

       deriv of Softmax -> softmax - ground truth
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = [1]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, weak_target):
        log_p = (probs[:, self.idc, ...] + 1e-10).log()

        one_hot_target = class2one_hot(weak_target, 2)
        mask = one_hot_target[:, self.idc, ...].type(torch.float32)

        loss = - einsum(f"bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


# ######## ------ Size loss function  (naive way) ---------- ###########
# --- This function will push the prediction to be close ot sizeGT ---#
class Size_Loss_naive():
    """
        Behaviour not exactly the same ; original numpy code used thresholding.
        Not quite sure it will have an inmpact or not
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, input):
        assert simplex(input)

        pred_size = einsum("bcwh->bc", input)[:, 1]
        target_size = 7845

        loss = (pred_size - target_size) ** 2

        return loss.mean() / 100
