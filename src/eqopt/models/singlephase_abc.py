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


    def prepare_for_loss(self) -> None:
        """Prepare model state before a fresh loss evaluation."""
        return None


    def project_composition(
        self,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        """Return a normalized phase composition, optionally projected by subclasses."""
        values = {
            element: float(comp.get(element, 0.0))
            for element in self.elements
        }
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError(
                f"Composition for phase {self.phase_name} must have positive sum."
            )
        return {
            element: value / total
            for element, value in values.items()
        }


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
