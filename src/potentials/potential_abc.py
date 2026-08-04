"""Abstract interfaces for torch-based atomistic potentials."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class TorchPotential(nn.Module, ABC):
    """Base class for potentials usable by ``TorchPotentialCalculator``.

    Subclasses are expected to expose ``dtype`` and ``device`` attributes.
    """

    @property
    @abstractmethod
    def cutoff(self) -> float:
        """Neighbor-list cutoff in Angstrom."""


    @abstractmethod
    def types_from_symbols(self, symbols: list[str] | tuple[str, ...]) -> np.ndarray:
        """Map atomic symbols to integer type ids used by the potential."""


    @abstractmethod
    def energy(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        types: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        """Return total potential energy for a periodic neighbor list."""


    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        types: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        return self.energy(positions, cell, types, i, j, shifts)
