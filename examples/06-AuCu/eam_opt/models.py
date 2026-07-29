from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.constants import Avogadro, electron_volt

from eqopt.dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from eqopt.models.shared import get_tensor_mu, normalize_and_order_composition
from eqopt.models.singlephase_abc import ThermodynamicModel
from eqopt.utilities import R

from .calc import TorchEAMFSCalculator
from .eam import EAMFSModule
from .utilities import get_composition_for_ase_atoms
from ase.filters import FrechetCellFilter
from ase.optimize import GoodOldQuasiNewton


@dataclasses.dataclass(frozen=True)
class BinaryGrandPotentialConfig:
    n_samples: int = 128
    n_steps: int = 20
    delta: float = 0.1
    use_softmin: bool = True
    softmin_tau: float | None = None
    eps: float = 1.0e-10


def _format_tdb_value(value: torch.Tensor | float) -> str:
    value = torch.as_tensor(value).detach().cpu().reshape(())
    return f"{float(value):+.8e}"


def _composition_tensor(
    composition: Mapping[str, float] | Sequence[float],
    elements: Sequence[str],
) -> torch.Tensor:
    if isinstance(composition, Mapping):
        values = [composition.get(element, 0.0) for element in elements]
    else:
        values = list(composition)
        if len(values) != len(elements):
            raise ValueError(
                f"Expected {len(elements)} composition values, got {len(values)}."
            )
    tensor = torch.as_tensor(values, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    tensor = tensor.clamp_min(1.0e-12)
    return tensor / tensor.sum().clamp_min(1.0e-12)


class CompPhase(ThermodynamicModel):
    """A line-compound model parameterized by one energy correction."""

    def __init__(
        self,
        phase_name: str,
        composition: Mapping[str, float],
        reference_energy: float,
        *,
        elements: Sequence[str] | None = None,
        correction_init: float = 0.0,
    ) -> None:
        elements = tuple(sorted(composition if elements is None else elements))
        super().__init__(phase_name, elements)
        normalized_composition = _composition_tensor(composition, elements)
        self.register_buffer(
            "composition",
            normalized_composition,
            persistent=True,
        )
        self.register_buffer(
            "reference_energy",
            torch.as_tensor(reference_energy, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE),
            persistent=True,
        )
        self.correction = torch.nn.Parameter(
            torch.as_tensor(correction_init, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        )

    @property
    def energy(self) -> torch.Tensor:
        return self.reference_energy + self.correction

    @property
    def composition_mapping(self) -> Mapping[str, float]:
        return {
            element: float(self.composition[index].detach().cpu())
            for index, element in enumerate(self.elements)
        }

    def project_composition(
        self,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        target = normalize_and_order_composition(comp, self.elements)
        if not torch.allclose(target, self.composition, atol=tol, rtol=0.0):
            raise ValueError(
                f"{self.phase_name} has fixed composition "
                f"{self.composition.detach().cpu().tolist()}; got "
                f"{target.detach().cpu().tolist()}."
            )
        return {
            element: float(self.composition[index].detach().cpu())
            for index, element in enumerate(self.elements)
        }

    def gibbs_energy_per_molar_atom(
        self,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        self.project_composition(comp)
        return self.energy

    def grand_potential_per_molar_atom(
        self,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        mu_tensor = get_tensor_mu(mu, self.elements)
        return self.energy - self.composition @ mu_tensor

    def get_tdb_str(self) -> str:
        multiplicities = " ".join(
            f"{float(x):g}" for x in self.composition.detach().cpu()
        )
        site_array = ":".join(self.elements)
        lines = [
            f"phase {self.phase_name} % {len(self.elements)} {multiplicities} !",
            f"constituent {self.phase_name} : {site_array} : !",
            (
                f"parameter g({self.phase_name},{site_array};0) "
                f"100 {_format_tdb_value(self.energy)}; 3000 N !"
            ),
        ]
        return "\n".join(lines).upper()


class EAMCompPhase(ThermodynamicModel):
    """A line-compound model whose energy comes from an EAM structure."""
    f_max: float = 1.0e-3
    step_max: int = 500
    def __init__(
        self,
        phase_name: str,
        atoms: Atoms,
        eam_model: EAMFSModule,
        *,
        composition: Mapping[str, float] | None = None,
        elements: Sequence[str] | None = None,
        hydro: bool = False,
    ) -> None:
        composition = (
            get_composition_for_ase_atoms(atoms)
            if composition is None
            else composition
        )
        elements = tuple(sorted(composition if elements is None else elements))
        super().__init__(phase_name, elements)
        self.eam_model = eam_model
        self.calculator = TorchEAMFSCalculator(self.eam_model)
        self.initial_atoms = atoms.copy()
        self.relaxed_atoms = atoms.copy()
        self.initial_atoms.pbc = True
        self.relaxed_atoms.pbc = True
        self.relaxed_atoms.calc = self.calculator
        self.hydro = hydro
        self._energy_cache: torch.Tensor | None = None

        self.register_buffer(
            "composition",
            _composition_tensor(composition, elements),
            persistent=True,
        )

    @property
    def composition_mapping(self) -> Mapping[str, float]:
        return {
            element: float(self.composition[index].detach().cpu())
            for index, element in enumerate(self.elements)
        }

    @property
    def energy(self) -> torch.Tensor:
        if self._energy_cache is None:
            self._energy_cache = self._energy_from_relaxed_atoms()
        return self._energy_cache

    def prepare_for_loss(self) -> None:
        self.calculator.reset() 
        # the above is necessary so that cached value is not used
        self.relaxed_atoms.calc = self.calculator
        cell_filter = FrechetCellFilter(self.relaxed_atoms, hydrostatic_strain=self.hydro)
        optimizer = GoodOldQuasiNewton(cell_filter, logfile=None)
        optimizer.run(fmax=self.f_max, steps=self.step_max)
        self._energy_cache = None

    def project_composition(
        self,
        comp: Mapping[str, float],
        *,
        tol: float = 1.0e-3,
    ) -> Mapping[str, float]:
        target = normalize_and_order_composition(comp, self.elements)
        if not torch.allclose(target, self.composition, atol=tol, rtol=0.0):
            raise ValueError(
                f"{self.phase_name} has fixed composition "
                f"{self.composition.detach().cpu().tolist()}; got "
                f"{target.detach().cpu().tolist()}."
            )
        return self.composition_mapping

    def gibbs_energy_per_molar_atom(
        self,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        self.project_composition(comp)
        return self.energy

    def grand_potential_per_molar_atom(
        self,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        mu_tensor = get_tensor_mu(mu, self.elements)
        return self.energy - self.composition @ mu_tensor

    def get_tdb_str(self) -> str:
        multiplicities = " ".join(
            f"{float(x):g}" for x in self.composition.detach().cpu()
        )
        site_array = ":".join(self.elements)
        lines = [
            f"phase {self.phase_name} % {len(self.elements)} {multiplicities} !",
            f"constituent {self.phase_name} : {site_array} : !",
            (
                f"parameter g({self.phase_name},{site_array};0) "
                f"100 {_format_tdb_value(self.energy)}; 3000 N !"
            ),
        ]
        return "\n".join(lines).upper()

    def _energy_from_relaxed_atoms(self) -> torch.Tensor:
        atoms = self.relaxed_atoms
        positions = torch.as_tensor(
            atoms.get_positions(),
            dtype=self.eam_model.dtype,
            device=self.eam_model.device,
        )
        cell = torch.as_tensor(
            atoms.cell.array,
            dtype=self.eam_model.dtype,
            device=self.eam_model.device,
        )
        types = torch.as_tensor(
            self.eam_model.types_from_symbols(atoms.get_chemical_symbols()),
            dtype=torch.long,
            device=self.eam_model.device,
        )
        i_np, j_np, shifts_np = neighbor_list("ijS", atoms, self.eam_model.cutoff)
        i = torch.as_tensor(i_np, dtype=torch.long, device=self.eam_model.device)
        j = torch.as_tensor(j_np, dtype=torch.long, device=self.eam_model.device)
        shifts = torch.as_tensor(
            shifts_np,
            dtype=self.eam_model.dtype,
            device=self.eam_model.device,
        )
        energy_ev = self.eam_model.energy(positions, cell, types, i, j, shifts)
        return energy_ev / len(atoms) * electron_volt * Avogadro


class SolutionPhase(ThermodynamicModel):
    """Binary solution model optimized through structure-energy corrections."""

    def __init__(
        self,
        phase_name: str,
        entries: Sequence[CompPhase],
        *,
        config: BinaryGrandPotentialConfig | None = None,
    ) -> None:
        if len(entries) < 3:
            raise ValueError(
                "SolutionPhase requires at least three structure entries."
            )
        elements = tuple(sorted(set().union(*(entry.elements for entry in entries))))
        if len(elements) != 2:
            raise ValueError("SolutionPhase currently supports binary phases only.")

        elements = tuple(elements)
        super().__init__(phase_name, elements)
        self.n_interaction_terms = len(entries) - 2
        self.config = config or BinaryGrandPotentialConfig()
        self.entries = torch.nn.ModuleList(entries)

        compositions = torch.stack(
            [
                _composition_tensor(entry.composition_mapping, self.elements)
                for entry in self.entries
            ],
            dim=0,
        )
        design = self._redlich_design_matrix(compositions)
        if design.shape[0] != design.shape[1]:
            raise ValueError(
                "SolutionPhase expects exactly enough entries to determine "
                "the pure-element and Redlich interaction parameters."
            )
        self.register_buffer("structure_compositions", compositions, persistent=True)
        self.register_buffer("fit_matrix", torch.linalg.pinv(design), persistent=True)

    def prepare_for_loss(self) -> None:
        for entry in self.entries:
            entry.prepare_for_loss()

    @property
    def structure_energies(self) -> torch.Tensor:
        return torch.stack(
            [entry.energy.reshape(()) for entry in self.entries],
            dim=0,
        )

    def redlich_parameters(self) -> torch.Tensor:
        return self.fit_matrix @ self.structure_energies

    def _redlich_design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[..., 0]
        x1 = x[..., 1]
        interaction_columns = [
            x0 * x1 * (x0 - x1) ** order
            for order in range(self.n_interaction_terms)
        ]
        return torch.stack([x0, x1, *interaction_columns], dim=-1)

    def _enthalpy_from_composition_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._redlich_design_matrix(x) @ self.redlich_parameters()

    def gibbs_energy_per_molar_atom(
        self,
        comp: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        temperature = torch.as_tensor(
            temperature,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        ).reshape(())
        x = normalize_and_order_composition(comp, self.elements)
        enthalpy = self._enthalpy_from_composition_tensor(x)
        entropy = R * temperature * (x * x.clamp_min(self.config.eps).log()).sum(dim=-1)
        return enthalpy + entropy

    def grand_potential_per_molar_atom(
        self,
        mu: Mapping[str, float],
        temperature: float,
        runtime_data: Any = None,
    ) -> torch.Tensor:
        config = self.config
        temperature = torch.as_tensor(
            temperature,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        ).reshape(())
        mu_tensor = get_tensor_mu(mu, self.elements)
        y = self._composition_grid(config.n_samples, config.eps)

        if config.softmin_tau is None:
            tau = 1.0 if y.shape[0] == 1 else 1.0 / torch.log(
                torch.as_tensor(float(y.shape[0]), device=y.device, dtype=y.dtype)
            )
        else:
            tau = float(config.softmin_tau)

        if config.n_steps > 0:
            y = self._optimize_grand_potential_y(
                y,
                mu_tensor.detach(),
                temperature,
                n_steps=config.n_steps,
                delta=config.delta,
                eps=config.eps,
            )

        values = self._grand_potential_values(y.detach(), mu_tensor, temperature)
        if config.use_softmin:
            return -tau * torch.logsumexp(-values / tau, dim=0)
        return torch.min(values)

    def _composition_grid(self, n_samples: int, eps: float) -> torch.Tensor:
        x0 = torch.linspace(
            eps,
            1.0 - eps,
            int(n_samples),
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        return torch.stack([x0, 1.0 - x0], dim=-1)

    def _grand_potential_values(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        enthalpy = self._enthalpy_from_composition_tensor(x)
        entropy = R * temperature * (x * x.clamp_min(self.config.eps).log()).sum(dim=-1)
        return enthalpy + entropy - x @ mu

    def _optimize_grand_potential_y(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        temperature: torch.Tensor,
        *,
        n_steps: int,
        delta: float,
        eps: float,
    ) -> torch.Tensor:
        y = y.detach()
        eta: torch.Tensor | None = None
        with torch.enable_grad():
            for _ in range(int(n_steps)):
                y = y.detach().requires_grad_(True)
                values = self._grand_potential_values(y, mu, temperature)
                grad = torch.autograd.grad(values.sum(), y)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                centered_grad = grad - (y * grad).sum(dim=-1, keepdim=True)
                if eta is None:
                    scale = (y * centered_grad.abs()).amax(dim=-1, keepdim=True)
                    eta = delta / scale.clamp_min(eps)
                y = F.softmax((y.clamp_min(eps).log() - eta * centered_grad), dim=-1)
        return y.detach()

    def get_tdb_str(self) -> str:
        params = self.redlich_parameters().detach()
        lines = [
            f"phase {self.phase_name} % 1 1.0 !",
            f"constituent {self.phase_name} : {','.join(self.elements)} : !",
        ]
        for element, value in zip(self.elements, params[:2], strict=True):
            lines.append(
                f"parameter g({self.phase_name},{element};0) "
                f"100 {_format_tdb_value(value)}; 3000 N !"
            )
        for order, value in enumerate(params[2:]):
            lines.append(
                f"parameter g({self.phase_name},{','.join(self.elements)};{order}) "
                f"100 {_format_tdb_value(value)}; 3000 N !"
            )
        return "\n".join(lines).upper()
