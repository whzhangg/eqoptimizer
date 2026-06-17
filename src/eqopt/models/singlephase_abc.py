from abc import ABC, abstractmethod
from typing import Sequence, Mapping

from torch import Tensor, nn


class ThermodynamicModel(nn.Module, ABC):
    """Base class for differentiable thermodynamic phase models."""

    def __init__(self, phase_name: str, elements: Sequence[str]) -> None:
        super().__init__()
        self.phase_name = phase_name
        self.elements = tuple(elements) # external elements without vacancy


    @abstractmethod
    def gibbs_energy_per_molar_atom(self, comp: Mapping[str, float], temperature: float) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""


    def forward(self, comp, temperature: float) -> Tensor:
        return self.gibbs_energy_per_molar_atom(comp, temperature)


    @abstractmethod
    def grand_potential_per_molar_atom(
        self,
        mu: Mapping[str, float], 
        temperature: float, 
        *,
        use_softmin: bool = True,
        tau: float | None = None, 
        n_samples_each_side = 64,
        n_steps: int = 6,
        delta: float = 0.3,
        max_step_factor: float = 1.5
    ) -> Tensor:
        """Return phase grand potential at chemical potential and temperature."""

