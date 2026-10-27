from collections.abc import Collection, Mapping
from typing import Any
from pathlib import Path
from torch import Tensor
from torch import nn
import torch

from ..dtype import DEFAULT_DEVICE
from ..phase import PhaseID
from .system_abc import ThermodynamicSystem
from .singlephase_abc import ThermodynamicModel


class EnsembleSystem(ThermodynamicSystem):

    def __init__(self,
        models_dict: Mapping[PhaseID, ThermodynamicModel],
        excluded_phase_ids: Collection[PhaseID] = (),
    ):
        phase_ids = tuple(models_dict)
        elements = set()
        for phid in phase_ids:
            elements |= set(phid.elements)

        super().__init__(phase_ids, elements)
        self.excluded_phase_ids = frozenset(excluded_phase_ids)
        self._phase_id_to_key = {}
        key_to_model = {}
        for iphase, (phid, model) in enumerate(models_dict.items()):
            key = f'phase_{iphase}'
            self._phase_id_to_key[phid] = key
            key_to_model[key] = model

        self.key_to_model = nn.ModuleDict(dict(key_to_model))


    @property
    def equilibrium_phase_ids(self) -> tuple[PhaseID, ...]:
        return tuple(
            phase_id
            for phase_id in self.phase_ids
            if phase_id not in self.excluded_phase_ids
        )


    def get_model_by_phase_id(self, phase_id: PhaseID) -> ThermodynamicModel:
        return self.key_to_model[self._get_phase_key(phase_id)]


    def prepare_for_loss(self) -> None:
        for model in self.key_to_model.values():
            model.prepare_for_loss()


    def create_runtime_data_for_phase(self, phase_id: PhaseID) -> Any:
        return self.get_model_by_phase_id(phase_id).create_runtime_data()


    def _get_phase_key(self, phase_id: PhaseID) -> str:
        try:
            return self._phase_id_to_key[phase_id]
        except KeyError:
            raise KeyError(f"Unknown phase id: {phase_id}") from None


    def project_composition_for_phase(
        self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        model = self.key_to_model[self._get_phase_key(phase_id)]
        return model.project_composition(comp, tol=tol)


    def gibbs_energy_per_molar_atom_for_phase(self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""
        model = self.key_to_model[self._get_phase_key(phase_id)]
        return model.gibbs_energy_per_molar_atom(comp, temperature, runtime_data)


    def grand_potential_per_molar_atom_for_phase(self,
        phase_id: PhaseID,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        model = self.key_to_model[self._get_phase_key(phase_id)]
        return model.grand_potential_per_molar_atom(mu, temperature, runtime_data)


    def save_model_to_pt(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)


    @classmethod
    def from_pt(cls, path: str | Path) -> "EnsembleSystem":
        model = torch.load(
            Path(path),
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__} checkpoint, got "
                f"{type(model).__name__}."
            )
        return model
