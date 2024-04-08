#!/usr/bin/env python3

import torch as t
from tqdm import tqdm
from .loss import SquareLoss

def detach_loss(loss):
    """Detach a torch loss result so it is not part of the autograd graph

    Args:
        loss (tensor or float): tensor with single float

    Returns:
        loss (float)
    """
    return float(loss.detach().cpu()) if isinstance(loss, t.Tensor) else loss

def gd(f, y, model, num_iterations=100,
       loss_fns=[SquareLoss()], lr=1e-1, optimizer=t.optim.Adam,
       optim_args={}, progress_bar=True
       ):
    """Gradient descent to minimize loss function

    Minimizes sum of weighted loss functions with respect to model coefficients:
    e.g. `lam1 * loss_fn1(f, y, d, coeffs) + lam2 * loss_fn2(f, y, d, coeffs) + ...`

    Args:
        f (Forward): forward operator with pytorch autograd support
        y (tensor): measurement stack
        model (science.model.Model): initialized model
        loss_fns (list[science.Loss]): custom loss functions which
            accept (f, y, density, coeffs) as args
        num_iterations (int): number of gradient descent iterations
        lr (float): learning rate
        optimizer (pytorch Optimizer): optimizer.  optional.  defaults to 'Adam'
        optim_args (dict): optional optimizer arguments

    Returns:
        coeffs (tensor): retrieved coeffs with smallest loss.  shape `model.coeffs_shape`
        y (tensor): retrieved coeffs passed through model and forward operator: f(model(coeffs))
        losses (dict): loss for each loss function at every iteration
    """

    if y is not None:
        y.requires_grad_()

    coeffs = t.ones(
        model.coeffs_shape,
        requires_grad=True,
        device=f.device,
        dtype=t.float64
    )

    best_loss = float('inf')
    best_coeffs = None

    # coeffs_log = []

    optimizer = optimizer([coeffs], lr=lr, **optim_args)
    # initialize empty list for logging loss values each iteration
    losses = {loss_fn: [] for loss_fn in loss_fns}
    # perform requested number of iterations
    for _ in (pbar := tqdm(range(num_iterations), disable=not progress_bar)):
        optimizer.zero_grad()

        density = model(coeffs)

        fidelity = regularizer = 0
        for loss_fn in loss_fns:
            loss = loss_fn(f, y, density, coeffs)
            if loss_fn.use_grad:
                if loss_fn.fidelity:
                    fidelity += loss
                else:
                    regularizer += loss
            # log the loss
            losses[loss_fn].append(detach_loss(loss))

        pbar.set_description(f'F:{fidelity:.1e} R:{regularizer:.1e}')

        tot_loss = fidelity + regularizer
        # save the reconstruction with the lowest loss
        if tot_loss < best_loss:
            best_coeffs = coeffs

        tot_loss.backward(retain_graph=True)
        optimizer.step()

        # do coeffs projections after gradient step
        if hasattr(model, 'proj'):
            coeffs.data = model.proj(coeffs)

        # coeffs_log.append(coeffs.detach().cpu())
    # losses['coeffs'] = coeffs_log

    y_result = f(model(best_coeffs))
    return best_coeffs, y_result, losses