"""ASE calculator for torch-based atomistic potentials."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from .potential_abc import TorchPotential


class TorchPotentialCalculator(Calculator):
    """ASE calculator backed by a ``TorchPotential``.

    Forces are computed as ``-dE / d(displacement)`` and stress is computed as
    ``dE / d(strain) / volume`` from zero-valued differentiable tensors.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        potential: TorchPotential,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = potential
        self.dtype = self.model.dtype
        self.device = self.model.device

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise RuntimeError("Calculator has no atoms")

        energy, forces, stress = self._energy_forces_stress(self.atoms)
        energy_value = float(energy.detach().cpu().item())
        self.results["energy"] = energy_value
        self.results["free_energy"] = energy_value
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["stress"] = stress.detach().cpu().numpy()

    def _energy_forces_stress(self, atoms):
        types_np = self.model.types_from_symbols(atoms.get_chemical_symbols())
        positions0 = torch.as_tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device
        )
        cell0 = torch.as_tensor(
            np.asarray(atoms.cell), dtype=self.dtype, device=self.device
        )
        displacement = torch.zeros_like(positions0, requires_grad=True)
        strain = torch.zeros(
            (3, 3), dtype=self.dtype, device=self.device, requires_grad=True
        )

        deformation = torch.eye(3, dtype=self.dtype, device=self.device) + strain
        positions = positions0 @ deformation + displacement
        cell = cell0 @ deformation

        i_np, j_np, shifts_np = neighbor_list("ijS", atoms, self.model.cutoff)
        i = torch.as_tensor(i_np, dtype=torch.long, device=self.device)
        j = torch.as_tensor(j_np, dtype=torch.long, device=self.device)
        shifts = torch.as_tensor(shifts_np, dtype=self.dtype, device=self.device)
        types = torch.as_tensor(types_np, dtype=torch.long, device=self.device)

        energy = self.model.energy(positions, cell, types, i, j, shifts)
        grad_displacement, grad_strain = torch.autograd.grad(
            energy, (displacement, strain), create_graph=False
        )
        forces = -grad_displacement
        stress = _full_3x3_to_voigt(grad_strain / atoms.get_volume())
        return energy, forces, stress


def _full_3x3_to_voigt(stress: torch.Tensor) -> torch.Tensor:
    return stress.reshape(-1)[
        torch.as_tensor([0, 4, 8, 5, 2, 1], device=stress.device)
    ]
