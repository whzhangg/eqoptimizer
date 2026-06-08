import math
import torch
from torch import Tensor
import typing
from scipy import constants

from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE

R = constants.R # 8.31446261815324
PRESSURE = constants.atm # 101325


def multi_simplex_samples_dirichlet(
    n_components: typing.List[int],
    n_samples_each_side: int,
    *,
    eps: float = 1.0e-8,
) -> Tensor:
    """dirichlet samples on the composition simplex.
    
    Parameters
    ---
    n_components: int
    n_samples_each_side: int
        it roughly gives the number of sample for each component.
    
    Return a tensor with shape (n_samples, n_comp) with each row sum up 
    to 1
    """
    max_n_components = max(n_components)
    nsamples_total = int(
        n_samples_each_side**(max_n_components-1) / math.factorial(max_n_components-1)
    )
    
    return torch.hstack(
        [
            simplex_samples_dirichlet(
                n_components=i,
                n_samples_total=nsamples_total,
                eps=eps
            ) for i in n_components
        ]
    )


def simplex_samples_dirichlet(
    n_components: int,
    *,
    n_samples_each_side: int | None = None,
    n_samples_total: int = None,
    eps: float = 1.0e-8,
) -> Tensor:
    """dirichlet samples on the composition simplex.
    
    Parameters
    ---
    n_components: int
    n_samples_each_side: int
        it roughly gives the number of sample for each component.
    
    Return a tensor with shape (n_samples, n_comp) with each row sum up 
    to 1
    """
    from torch.distributions.dirichlet import Dirichlet
    if n_samples_total is not None:
        nsamples_total = n_samples_total
    else:
        if n_samples_each_side is None:
            raise ValueError(
                "Either n_samples_each_side or n_samples_total must be supplied."
            )
        nsamples_total = int(
            n_samples_each_side**(n_components-1) / math.factorial(n_components-1)
        )
    sampler = Dirichlet(torch.ones(n_components, dtype=DEFAULT_TYPE, device=DEFAULT_DEVICE))
    samples = sampler.sample((nsamples_total,)).to(device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)

    samples = samples.clamp_min(eps)
    return samples / samples.sum(dim=-1, keepdim=True)
