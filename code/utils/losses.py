#!/usr/bin/env python3

from operator import add
from typing import Callable
from functools import reduce

import torch
import numpy as np
import torch.nn.functional as F
from torch import einsum
from torch import Tensor

from .utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


# ######## ------ Size loss function  (naive way) ---------- ###########
# --- This function will push the prediction to be close ot sizeGT ---#
class NaiveSizeLoss():
    """
    This one implement the naive quadratic penalty
    penalty = 0                  if a <= pred_size
              (a - pred_size)^2  otherwise
    """
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, bounds):
        assert simplex(pred_softmax)

        B, K, H, W = pred_softmax.shape
        assert bounds.shape == (B, K, 2)

        pred_size = einsum("bkwh->bk", pred_softmax)[:, self.idk]

        upper_bounds = bounds[:, self.idk, 1]
        lower_bounds = bounds[:, self.idk, 0]
        assert (upper_bounds >= 0).all() and (lower_bounds >= 0).all()

        # size < upper <==> size - upper < 0
        # lower < size <==> lower - size < 0
        loss = F.relu(pred_size - upper_bounds) ** 2 + F.relu(lower_bounds - pred_size) ** 2
        loss /= (W * H)

        return loss


# Quadratic penalty
class ParametrableQuadraticPenalty():
    """
    This one implement the naive quadratic penalty
    penalty = 0                  if a <= pred_size
              (a - pred_size)^2  otherwise
    """
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.function: Callable[[Tensor], Tensor] = kwargs["function"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, bounds):
        assert simplex(pred_softmax)

        B, K, H, W = pred_softmax.shape
        _, _, M, _ = bounds.shape
        assert bounds.shape == (B, K, M, 2)

        pred_fn = self.function(pred_softmax)[:, self.idk]

        upper_bounds = bounds[:, self.idk, :, 1]
        lower_bounds = bounds[:, self.idk, : 0]
        assert (upper_bounds >= 0).all() and (lower_bounds >= 0).all()

        # pred < upper <==> pred - upper < 0
        # lower < pred <==> lower - pred < 0
        loss = F.relu(pred_fn - upper_bounds) ** 2 + F.relu(lower_bounds - pred_fn) ** 2
        loss /= (W * H)

        return loss.sum()


# With a log-barrier in place of quadratic penalty
class ParametrableLogBarrier():
    """
    This one implement the naive quadratic penalty
    penalty = 0                  if a <= pred_size
              (a - pred_size)^2  otherwise
    """
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.function: Callable[[Tensor], Tensor] = kwargs["function"]
        self.t: float = 1
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __barrier__(self, z: Tensor) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / self.t**2:
            res = - torch.log(-z) / self.t
        else:
            res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        # # print(res)

        return res

    def __call__(self, pred_softmax, bounds):
        assert simplex(pred_softmax)

        B, K, H, W = pred_softmax.shape
        _, _, M, _ = bounds.shape
        assert bounds.shape == (B, K, M, 2)

        pred_fn = self.function(pred_softmax)[:, self.idk].view(B, len(self.idk), M)

        upper_bounds = bounds[:, self.idk, :, 1]
        lower_bounds = bounds[:, self.idk, :, 0]

        assert pred_fn.shape == upper_bounds.shape == lower_bounds.shape

        # pred < upper <==> pred - upper < 0
        # lower < pred <==> lower - pred < 0
        upper_part: Tensor = reduce(add, (self.__barrier__(z) for z in (pred_fn - upper_bounds).flatten()))
        lower_part: Tensor = reduce(add, (self.__barrier__(z) for z in (lower_bounds - pred_fn).flatten()))

        loss: Tensor = upper_part + lower_part
        loss /= (W * H)

        return loss
