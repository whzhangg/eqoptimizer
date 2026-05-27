import math
import torch
from torch import Tensor
import typing
from scipy import constants

from .dtype import TORCH_FLOAT

R = constants.R # 8.31446261815324
PRESSURE = constants.atm # 101325


def as_float_tensor(value, *, device=None, dtype=None) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype or TORCH_FLOAT)


def multi_simplex_samples_dirichlet(
    n_components: typing.List[int],
    n_samples_each_side: int = 128,
    *,
    device=None,
    dtype=None,
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
                device=device,
                dtype=dtype,
                eps=eps
            ) for i in n_components
        ]
    )

def simplex_samples_dirichlet(
    n_components: int,
    n_samples_each_side: int = 128,
    *,
    n_samples_total: int = None,
    device=None,
    dtype=None,
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
        nsamples_total = int(
            n_samples_each_side**(n_components-1) / math.factorial(n_components-1)
        )
    dtype = dtype or TORCH_FLOAT
    sampler = Dirichlet(torch.ones(n_components, dtype=dtype, device=device))
    samples = sampler.sample((nsamples_total,)).to(device=device, dtype=dtype)

    samples = samples.clamp_min(eps)
    return samples / samples.sum(dim=-1, keepdim=True)
