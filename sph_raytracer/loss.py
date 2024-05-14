"""Optimizer loss functions

The classes in this module serve as loss functions for the iterative reconstruction methods in `retrieval.py`.

The class should be initialized by the user and then passed to the reconstruction method which will call `.compute()`
on the loss function and try to minimize the weighted sum of all provided losses.

Weights are stored internally in the `lam` parameter, which the user may set by multiplying the
initialized loss object with a float or by providing a `lam` kwarg on initialization.
"""

from dataclasses import dataclass, field
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

    # projection_mask: t.Tensor = 1
    # volume_mask: t.Tensor = 1
    # lam: float = 1
    # fidelity: bool = False
    # use_grad: bool = True
    projection_mask: t.Tensor = field(kw_only=True, default=1)
    volume_mask: t.Tensor = field(kw_only=True, default=1)
    lam: float = field(kw_only=True, default=1)
    fidelity: bool = field(kw_only=True, default=False)
    use_grad: bool = field(kw_only=True, default=True)

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


@dataclass(unsafe_hash=True, repr=False)
class SquareLoss(Loss):
    """Standard mean L2 loss"""

    fidelity: bool = True

    def compute(self, f, y, d, c):
        """"""
        result = t.mean(self.projection_mask * (y - f(d * self.volume_mask))**2)
        return result


@dataclass(unsafe_hash=True, repr=False)
class SquareRelLoss(Loss):
    """Loss as mean percent error"""

    fidelity = True

    def compute(self, f, y, d, c):
        """"""
        obs = f(d * self.volume_mask)

        # rel_err = (y - obs) / y
        # rel_err = rel_err.nan_to_num() * self.projection_mask

        zero_mask = (y != 0)
        rel_err = t.zeros_like(y)
        rel_err[zero_mask] = (y - obs)[zero_mask] / y[zero_mask]

        return t.mean((self.projection_mask * rel_err)**2)


@dataclass(unsafe_hash=True, repr=False)
class CheaterLoss(Loss):
    """L2 loss directly over density ground truth"""

    def __init__(self, density_truth, **kwargs):
        self.density_truth = density_truth

        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""
        return t.mean(self.volume_mask * (d - self.density_truth)**2)


@dataclass(unsafe_hash=True, repr=False)
class NegRegularizer(Loss):
    """Mean of negative voxels"""
    def compute(self, f, y, d, c):
        """"""
        return t.mean(t.abs(self.volume_mask * d.clip(max=0)))


@dataclass(unsafe_hash=True, repr=False)
class NegSumRegularizer(Loss):
    """Sum of negative voxels"""
    def compute(self, f, y, d, c):
        """"""
        return t.sum(t.abs(self.volume_mask * d.clip(max=0)))