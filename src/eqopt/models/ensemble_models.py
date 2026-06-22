from collections.abc import MutableMapping, Mapping, Sequence
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

    def __init__(self, models_dict: Mapping[PhaseID, ThermodynamicModel]):
        phase_ids = tuple(models_dict)
        elements = set()
        for phid in phase_ids:
            elements |= set(phid.elements)

        super().__init__(phase_ids, elements)
        self._phase_id_to_key = {}
        key_to_model = {}
        for iphase, (phid, model) in enumerate(models_dict.items()):
            key = f'phase_{iphase}'
            self._phase_id_to_key[phid] = key
            key_to_model[key] = model

        self.key_to_model = nn.ModuleDict(dict(key_to_model))


    def get_model_by_phase_id(self, phase_id: PhaseID) -> ThermodynamicModel:
        return self.key_to_model[self._get_phase_key(phase_id)]


    def _get_phase_key(self, phase_id: PhaseID) -> str:
        try:
            return self._phase_id_to_key[phase_id]
        except KeyError:
            raise KeyError(f"Unknown phase id: {phase_id}") from None


    def gibbs_energy_per_molar_atom_for_phase(self,
        phase_id: PhaseID,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""
        model = self.key_to_model[self._get_phase_key(phase_id)]
        data = self._runtime_data_for_phase(model, runtime_data, phase_id)
        return model.gibbs_energy_per_molar_atom(comp, temperature, data)


    def grand_potential_per_molar_atom_for_phase(self,
        phase_id: PhaseID,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> Tensor:
        model = self.key_to_model[self._get_phase_key(phase_id)]
        data = self._runtime_data_for_phase(model, runtime_data, phase_id)
        return model.grand_potential_per_molar_atom(mu, temperature, data)


    @staticmethod
    def _runtime_data_for_phase(
        model: ThermodynamicModel,
        runtime_data: Any,
        phase_id: PhaseID,
    ) -> Any:
        if runtime_data is None:
            return None
        if not isinstance(runtime_data, MutableMapping):
            raise TypeError("runtime_data must be a mutable mapping or None.")
        if phase_id not in runtime_data:
            runtime_data[phase_id] = model.create_runtime_data()
        return runtime_data[phase_id]


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
