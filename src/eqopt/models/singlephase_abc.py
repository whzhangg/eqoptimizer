from abc import ABC, abstractmethod
from typing import Any, Sequence, Mapping

from torch import Tensor, nn


class ThermodynamicModel(nn.Module, ABC):
    """Base class for differentiable thermodynamic phase models."""

    def __init__(self, phase_name: str, elements: Sequence[str]) -> None:
        super().__init__()
        self.phase_name = phase_name
        self.elements = tuple(elements) # external elements without vacancy


    def create_runtime_data(self) -> Any:
        """Return model-specific transient data for one optimization context."""
        return None


    @abstractmethod
    def gibbs_energy_per_molar_atom(
        self,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""


    def forward(self, comp, temperature: float, runtime_data: Any = None) -> Tensor:
        return self.gibbs_energy_per_molar_atom(comp, temperature, runtime_data)


    @abstractmethod
    def grand_potential_per_molar_atom(
        self,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return phase grand potential at chemical potential and temperature."""
