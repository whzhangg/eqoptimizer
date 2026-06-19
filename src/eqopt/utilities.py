import math
import numpy as np
from scipy.optimize import linprog
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


def hit_and_run_sampling(
    C: torch.Tensor,
    d: torch.Tensor,
    n_samples_final: int = 64,
    *,
    reduce_by_fps: bool = True,
    n_samples_to_sample: int =1000
) -> Tensor:
    """
    monte carlo sampling of positive numbers that satisfy a set of
    linear constraints. For CEF, constraints are:
    y on each sublattice sum to 1,
    all compositions are reproduced (one is redundent)
    """
    #if not isinstance(C, torch.Tensor):
    #    raise TypeError("C must be a torch.Tensor.")
    #if not isinstance(d, torch.Tensor):
    #    raise TypeError("d must be a torch.Tensor.")
    #if C.ndim != 2:
    #    raise ValueError(f"C must be a matrix, got shape {tuple(C.shape)}.")
    #if d.ndim != 1 or d.shape[0] != C.shape[0]:
    #    raise ValueError(
    #        "d must be a vector with length matching the number of rows in C; "
    #        f"got C shape {tuple(C.shape)} and d shape {tuple(d.shape)}."
    #    )
    #if d.device != C.device:
    #    raise ValueError("C and d must be on the same device.")
    #if d.dtype != C.dtype:
    #    raise ValueError("C and d must have the same dtype.")

    device = C.device
    dtype = C.dtype

    C_numpy = C.detach().cpu().numpy()
    d_numpy = d.detach().cpu().numpy()
    res = linprog(
        c=np.zeros(C.shape[1]),
        A_eq=C_numpy,
        b_eq=d_numpy,
        bounds=[(0, None)] * C.shape[1],
        method="highs",
    )
    if not res.success:
        raise ValueError("No feasible point found.")

    x0 = torch.as_tensor(res.x, device=device, dtype=dtype)

    _, singular_values, vh = torch.linalg.svd(C, full_matrices=True)
    if singular_values.numel() == 0:
        rank = 0
    else:
        tolerance = (
            torch.finfo(dtype).eps
            * max(C.shape)
            * singular_values.amax()
        )
        rank = int((singular_values > tolerance).sum().item())
    null_basis = vh[rank:].mT.contiguous()

    if not reduce_by_fps:
        n_samples_to_sample = n_samples_final

    if null_basis.shape[-1] == 0:
        samples = x0.expand(n_samples_to_sample, -1).clone()
        if reduce_by_fps:
            return samples[:n_samples_final]
        return samples

    samples: list[torch.Tensor] = []
    x = x0
    eps = torch.finfo(dtype).eps
    direction_tol = 1.0e-14
    for _ in range(n_samples_to_sample):
        # random direction in null space
        u = torch.randn(
            null_basis.shape[-1],
            device=device,
            dtype=dtype,
        )
        v = null_basis @ u
        norm = torch.linalg.vector_norm(v)
        if norm <= eps:
            continue
        v = v / norm

        positive = v > direction_tol
        negative = v < -direction_tol
        if torch.any(positive):
            t_min = torch.max(-x[positive] / v[positive])
        else:
            t_min = torch.as_tensor(-torch.inf, device=device, dtype=dtype)
        if torch.any(negative):
            t_max = torch.min(-x[negative] / v[negative])
        else:
            t_max = torch.as_tensor(torch.inf, device=device, dtype=dtype)

        if not torch.isfinite(t_min) or not torch.isfinite(t_max) or t_min > t_max:
            raise RuntimeError("Hit-and-run produced an invalid line interval.")

        t = t_min + torch.rand((), device=device, dtype=dtype) * (t_max - t_min)
        x = x + t * v

        # clean numerical noise
        x = torch.where(x.abs() < 1.0e-14, torch.zeros_like(x), x)

        samples.append(x.clone())

    if not samples:
        samples_tensor = x0.expand(n_samples_to_sample, -1).clone()
    else:
        samples_tensor = torch.stack(samples, dim=0)

    if reduce_by_fps:
        idx = torch_fps_sampling(samples_tensor, n_samples_final)
        return samples_tensor[idx]
    else:
        return samples_tensor


def torch_fps_sampling(input: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Return farthest-point-sampling indices for a 2D tensor.

    The first point is selected randomly. Subsequent points maximize the
    squared distance to the closest already selected point.
    """
    #if input.ndim != 2:
    #    raise ValueError(f"input must be a 2D tensor, got shape {tuple(input.shape)}.")
    n_points = input.shape[0]
    if n_points == 0:
        raise ValueError("Cannot sample from an empty tensor.")
    #if n_samples <= 0:
    #    return torch.empty(0, device=input.device, dtype=torch.long)

    n_selected = min(int(n_samples), n_points)
    selected = torch.empty(n_selected, device=input.device, dtype=torch.long)

    first = torch.randint(n_points, (), device=input.device)
    selected[0] = first
    closest_distance = (input - input[first]).square().sum(dim=-1)

    for i in range(1, n_selected):
        next_index = torch.argmax(closest_distance)
        selected[i] = next_index
        new_distance = (input - input[next_index]).square().sum(dim=-1)
        closest_distance = torch.minimum(closest_distance, new_distance)

    return selected
