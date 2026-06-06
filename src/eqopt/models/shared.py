from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ..dtype import DEFAULT_DEVICE, DEFAULT_TYPE


def scalar_temperature(temperature) -> Tensor:
    """Return temperature as a scalar tensor and reject vectorized temperatures."""
    temperature = torch.as_tensor(temperature, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if temperature.numel() != 1:
        raise ValueError("Temperature must be a scalar.")
    return temperature.reshape(())


def get_tensor_mu(mu_dict: Mapping[str, float | Tensor], elements: Sequence[str]) -> Tensor:
    return torch.stack(
        [
            torch.as_tensor(mu_dict[ele], device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
            for ele in elements
        ],
        dim=0,
    )


def temperature_powers(
    temperature,
    orders: int | Sequence[int] | Tensor,
    temperature_ref: float,
) -> Tensor:
    """Return powers of scaled temperature, (T / T_ref)^n."""
    temperature = torch.as_tensor(temperature, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if isinstance(orders, int):
        powers = torch.arange(
            orders + 1,
            device=temperature.device,
            dtype=temperature.dtype,
        )
    elif isinstance(orders, Tensor):
        powers = orders.to(device=temperature.device, dtype=temperature.dtype)
    elif isinstance(orders, Sequence):
        powers = torch.as_tensor(
            orders, device=temperature.device, dtype=temperature.dtype
        )
    else:
        raise ValueError(f'orders should be int or sequence of int, get {orders}')
    return (temperature[..., None] / temperature_ref) ** powers


def normalize_and_order_composition(
    comp: Mapping[str, float | Tensor],
    elements: tuple[str, ...],
) -> Tensor:
    """Return normalized composition ordered according to model elements."""

    missing = tuple(ele for ele in elements if ele not in comp)
    if missing:
        raise ValueError(
            f"Composition is missing elements {missing}; expected {elements}."
        )
    columns = [
        torch.as_tensor(comp[ele], device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        for ele in elements
    ]
    x = torch.stack(torch.broadcast_tensors(*columns), dim=-1)
    x = x.clamp_min(1.0e-12)
    return x / x.sum(dim=-1, keepdim=True)
