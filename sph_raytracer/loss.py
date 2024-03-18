#!/usr/bin/env python3

from dataclasses import dataclass
import torch as t

@dataclass(unsafe_hash=True)
class Loss:
    """Loss function for tomographic retrieval

    Args:
        projection_mask (tensor): column densities to mask out when computing loss
        volume_mask (tensor): voxels to mask out when computing loss
        lam (float): loss function scaling
        fidelity (bool): whether this is the fidelity term (for plotting/display purposes)
        use_grad (bool): whether this loss function's gradient needs to be used in optimization

    Usage:
        gd(..., losses=[5 * MyLoss(), 3 * MyLoss2()], ...)
    """

    projection_mask = 1
    volume_mask = 1
    lam = 1
    fidelity = False
    use_grad = True

    # def __init__(self, projection_mask=1, volume_mask=1, lam=1, fidelity=False, use_grad=True):
    #     self.projection_mask = projection_mask
    #     self.volume_mask = volume_mask
    #     self.lam = lam
    #     self.fidelity = fidelity
    #     self.use_grad = use_grad

    def compute(self, f, y, d, c):
        """Compute loss

        Args:
            f (Forward): forward function. density→projections
            y (tensor): measurements.  shape must match `projection_mask`
            d (tensor): density to pass through forward function.
                shape must match `volume_mask`
            c (tensor): coefficients of shape model.coeffs_shape

        Returns:
            loss (float)
        """
        raise NotImplemented

    def __call__(self, f, y, d, c):
        """Compute loss, incorporating loss weight and whether pytorch grad is needed

        Args:
            f (Forward): forward function. density→projections
            y (tensor): measurements.  shape must match `projection_mask`
            d (tensor): density to pass through forward function.
                shape must match `volume_mask`
            c (tensor): coefficients of shape model.coeffs_shape

        Returns:
            loss (float or None)
        """
        if self.use_grad:
            result = self.compute(f, y, d, c)
        else:
            with t.no_grad():
                result = self.compute(f, y, d, c)
        return None if result is None else self.lam * result

    def __mul__(self, other):
        """Allow multiplying Loss object with scalar hyperparameter"""
        self.lam = other
        return self

    def __rmul__(self, other):
        """Allow multiplying Loss object with scalar hyperparameter"""
        return self.__mul__(other)

    def __repr__(self):
        return f'{self.lam:.0e} * {type(self).__name__}'
        # return f'{type(self).__name__}'


class SquareLoss(Loss):
    """Standard mean L2 loss"""

    fidelity = True

    def compute(self, f, y, d, c):
        """"""
        return t.mean(self.projection_mask * (y - f(d * self.volume_mask))**2)


class SquareRelLoss(Loss):
    """Loss as mean percent error"""

    fidelity = True

    def compute(self, f, y, d, c):
        """"""
        obs = f(d * self.volume_mask)
        rel_err = (y - obs) / obs
        rel_err = rel_err.nan_to_num() * self.projection_mask
        return t.mean(rel_err**2)

class CheaterLoss(Loss):
    """L2 loss directly over density ground truth"""

    def __init__(self, density_truth, **kwargs):
        self.density_truth = density_truth

        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""
        return t.mean(self.volume_mask * (d - self.density_truth)**2)

class NegRegularizer(Loss):
    """Mean of negative voxels"""
    def compute(self, f, y, d, c):
        """"""
        return t.mean(t.abs(self.volume_mask * d.clip(max=0)))

class NegSumRegularizer(Loss):
    """Sum of negative voxels"""
    def compute(self, f, y, d, c):
        """"""
        return t.sum(t.abs(self.volume_mask * d.clip(max=0)))