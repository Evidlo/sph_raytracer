"""Tomographic retrieval module

This module provides methods for performing tomographic retrievals from a set of measurements
"""

import torch as t
from tqdm import tqdm
from .loss import SquareLoss

def detach_loss(loss):
    """Detach a torch loss result so it is not part of the autograd graph.  Use this when
    keeping track of some oracle loss function (e.g. comparing against a known ground-truth)
    that is not used by the PyTorch optimizer

    Args:
        loss (tensor or float): tensor with single float

    Returns:
        loss (float)
    """
    return float(loss.detach().cpu()) if isinstance(loss, t.Tensor) else loss

def gd(f, y, model, coeffs=None, num_iterations=100,
       loss_fns=[SquareLoss()], optim=t.optim.Adam,
       progress_bar=True, **kwargs
       ):
    """Gradient descent to minimize loss function.  Instantiates and optimizes a set of coefficients
    for the given model with respect to provided loss functions

    Minimizes sum of weighted loss functions with respect to model coefficients:
    e.g. `loss_fn1(f, y, d, coeffs) + loss_fn2(f, y, d, coeffs) + ...`

    Use Ctrl-C to stop iterations early and return best result so far.

    Args:
        f (Forward): forward operator with pytorch autograd support
        y (tensor): measurement stack
        model (science.model.Model): initialized model
        coeffs (tensor): initial value of coeffs before optimizing.
            should have `requires_grad=True`.  defaults to `t.ones(model.coeffs_shape)`
        loss_fns (list[science.Loss]): custom loss functions which
            accept (f, y, density, coeffs) as args.  Losses are summed
        num_iterations (int): number of gradient descent iterations
        optim (pytorch Optimizer): optimizer.  optional.  defaults to 'Adam'
        **kwargs (dict): optional optimizer arguments

    Returns:
        coeffs (tensor): retrieved coeffs with smallest loss.  shape `model.coeffs_shape`
        y (tensor): retrieved coeffs passed through model and forward operator: f(model(coeffs))
        losses (dict[list[float]]): loss for each loss function at every iteration
    """

    if f.grid != model.grid:
        raise ValueError("f and model must have same grid")

    if y is not None:
        y.requires_grad_()

    if coeffs is None:
        coeffs = t.ones(
            model.coeffs_shape,
            requires_grad=True,
            # FIXME: why were we using f.device here?
            device=f.device,
            dtype=t.float64
        )

    best_loss = float('inf')
    best_coeffs = None

    optim = optim([coeffs], **kwargs)
    # initialize empty list for logging loss values each iteration
    losses = {loss_fn: [] for loss_fn in loss_fns}
    # perform requested number of iterations
    try:
        for _ in (pbar := tqdm(range(num_iterations), disable=not progress_bar)):
            optim.zero_grad()

            density = model(coeffs)

            fidelity = regularizer = oracle = 0
            for loss_fn in loss_fns:
                loss = loss_fn(f, y, density, coeffs)
                if loss_fn.use_grad:
                    if loss_fn.fidelity:
                        fidelity += loss
                    elif loss_fn.oracle:
                        oracle += loss
                    else:
                        regularizer += loss
                # log the loss
                losses[loss_fn].append(detach_loss(loss))

            pbar.set_description(f'F:{fidelity:.1e} R:{regularizer:.1e} O:{oracle:.1e}')

            tot_loss = fidelity + regularizer
            # save the reconstruction with the lowest loss
            if tot_loss < best_loss:
                best_coeffs = coeffs

            tot_loss.backward(retain_graph=True)
            optim.step()

            # do coeffs projections after gradient step
            if hasattr(model, 'proj'):
                coeffs.data = model.proj(coeffs)

    # allow user to stop iterations
    except KeyboardInterrupt:
        pass

    y_result = f(model(best_coeffs))
    return best_coeffs, y_result, losses
