from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Set, Collection
from torch import Tensor, nn

from ..phase import PhaseID


class ThermodynamicSystem(nn.Module, ABC):
    """provide thermodynamic potential for defines phases"""
    def __init__(self, phase_ids: Sequence[PhaseID], elements: Set[str]) -> None:
        super().__init__()
        self.phase_ids = tuple(phase_ids)
        self.elements = set(elements)


    def get_competing_phases(self, elements: Collection[str]) -> Sequence[PhaseID]:
        """return possible phases from a set of elements"""
        ele_set = set(elements)
        return [phase for phase in self.phase_ids if set(phase.elements) <= ele_set]

    
    def forward(self, phase_id: PhaseID, comp, temperature: float) -> Tensor:
        return self.gibbs_energy_per_molar_atom_for_phase(phase_id, comp, temperature)


    @abstractmethod
    def gibbs_energy_per_molar_atom_for_phase(self, 
        phase_id: PhaseID, 
        comp: Mapping[str, float], 
        temperature: float
    ) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""
    

    def get_gibbs_energy(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        temperature: float,
    ) -> Tensor:
        return self.gibbs_energy_per_molar_atom_for_phase(
            phase_id,
            comp,
            temperature,
        )


    @abstractmethod
    def grand_potential_per_molar_atom_for_phase(
        self,
        phase_id: PhaseID,
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


    def get_grand_potential(
        self,
        phase_id: PhaseID,
        mu: Mapping[str, float],
        temperature: float,
        *,
        use_softmin: bool = True,
        tau: float | None = None,
        n_samples_each_side=64,
        n_steps: int = 6,
        delta: float = 0.3,
        max_step_factor: float = 1.5,
    ) -> Tensor:
        return self.grand_potential_per_molar_atom_for_phase(
            phase_id,
            mu,
            temperature,
            use_softmin=use_softmin,
            tau=tau,
            n_samples_each_side=n_samples_each_side,
            n_steps=n_steps,
            delta=delta,
            max_step_factor=max_step_factor,
        )
