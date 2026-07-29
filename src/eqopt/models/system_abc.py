from abc import ABC, abstractmethod
from typing import Any, Sequence, Mapping, Set, Collection
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


    @abstractmethod
    def project_composition_for_phase(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        """Return a valid composition for a phase, correcting small input errors."""


    def project_composition(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        """return a valid composition if it is within tolerance from feasible ones"""
        return self.project_composition_for_phase(phase_id, comp, tol=tol)


    def prepare_for_loss(self) -> None:
        """Prepare system state before a fresh loss evaluation."""
        return None

    
    def forward(
        self,
        phase_id: PhaseID,
        comp,
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        return self.gibbs_energy_per_molar_atom_for_phase(
            phase_id, comp, temperature, runtime_data)


    @abstractmethod
    def gibbs_energy_per_molar_atom_for_phase(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""
    

    def get_gibbs_energy(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        return self.gibbs_energy_per_molar_atom_for_phase(
            phase_id,
            comp,
            temperature,
            runtime_data
        )


    @abstractmethod
    def grand_potential_per_molar_atom_for_phase(
        self,
        phase_id: PhaseID,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return phase grand potential at chemical potential and temperature."""


    def get_grand_potential(
        self,
        phase_id: PhaseID,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        return self.grand_potential_per_molar_atom_for_phase(
            phase_id,
            mu,
            temperature,
            runtime_data
        )
