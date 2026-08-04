"""PyTorch EAM/FS potential module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..potential_abc import TorchPotential
from .phraser import EAMFSData, read_eam_fs


class EAM(TorchPotential):
    """Torch module evaluating a LAMMPS/DYNAMO ``eam/fs`` potential."""
    def __init__(
        self,
        potential: str | Path | EAMFSData,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.potential = (
            read_eam_fs(potential)
            if isinstance(potential, (str, Path))
            else potential
        )
        self.dtype = dtype
        self.device = torch.device(device)
        self._symbol_to_type = {
            symbol: index for index, symbol in enumerate(self.potential.symbols)
        }
        self.register_buffer(
            "_embedding",
            torch.as_tensor(
                _lammps_eam_spline_coefficients(self.potential.embedding),
                dtype=dtype,
                device=self.device,
            ),
            persistent=True,
        )
        self.register_buffer(
            "_density",
            torch.as_tensor(
                _lammps_eam_spline_coefficients(self.potential.density),
                dtype=dtype,
                device=self.device,
            ),
            persistent=True,
        )
        self.register_buffer(
            "_rphi",
            torch.as_tensor(
                _lammps_eam_spline_coefficients(self.potential.rphi),
                dtype=dtype,
                device=self.device,
            ),
            persistent=True,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.potential.symbols

    @property
    def cutoff(self) -> float:
        return self.potential.cutoff

    @property
    def symbol_to_type(self) -> dict[str, int]:
        return dict(self._symbol_to_type)

    def types_from_symbols(self, symbols: list[str] | tuple[str, ...]) -> np.ndarray:
        try:
            return np.asarray([self._symbol_to_type[s] for s in symbols], dtype=int)
        except KeyError as exc:
            allowed = ", ".join(self.potential.symbols)
            raise ValueError(f"Element {exc.args[0]!r} is not in potential ({allowed})")

    def energy(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        types: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        vectors = positions[j] + shifts @ cell - positions[i]
        distances = torch.linalg.norm(vectors, dim=1)

        density_values = self._evaluate_r_table(
            self._density[types[i], types[j]], distances
        )
        atomic_density = torch.zeros(
            positions.shape[0], dtype=self.dtype, device=self.device
        ).index_add(0, i, density_values)
        embedding_energy = self._evaluate_rho_table(
            self._embedding[types], atomic_density
        ).sum()

        rphi_values = self._evaluate_r_table(
            self._rphi[types[i], types[j]], distances
        )
        safe_distances = distances.clamp_min(torch.finfo(self.dtype).eps)
        pair_energy = 0.5 * (rphi_values / safe_distances).sum()
        return embedding_energy + pair_energy

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

    def _evaluate_r_table(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return _evaluate_lammps_eam_spline(coeffs, x, self.potential.dr)

    def _evaluate_rho_table(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return _evaluate_lammps_eam_spline(coeffs, x, self.potential.drho)


@dataclass
class FineTunedEAMConfig:
    """Configuration for zero-centered radial EAM/FS corrections."""

    n_pair_basis: int = 8
    n_density_basis: int = 8
    r_min: float = 1.0
    r_max: float | None = None
    optimize_pair: bool = True
    optimize_density: bool = True
    pair_scale: float = 0.5
    density_scale: float = 0.05



class FineTunedEAM(EAM):
    """EAM/FS module with trainable radial corrections.

    The baseline embedding, density, and pair tables are read directly from the
    input ``eam.fs`` file.  The embedding functions are kept fixed.  Optional
    zero-initialized Chebyshev corrections are added to all pair potentials and
    all directed density functions.
    """

    def __init__(
        self,
        potential: str | Path | EAMFSData,
        *,
        config: FineTunedEAMConfig | None = None,
        pair_mask: torch.Tensor | np.ndarray | None = None,
        density_mask: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(potential, dtype=dtype, device=device)
        self.config = FineTunedEAMConfig() if config is None else config
        if self.config.n_pair_basis < 1:
            raise ValueError("n_pair_basis must be at least 1.")
        if self.config.n_density_basis < 1:
            raise ValueError("n_density_basis must be at least 1.")

        self.r_min = float(self.config.r_min)
        self.r_max = (
            float(self.cutoff)
            if self.config.r_max is None
            else float(self.config.r_max)
        )
        if not (0.0 <= self.r_min < self.r_max <= self.cutoff):
            raise ValueError(
                "Require 0 <= r_min < r_max <= potential cutoff for corrections."
            )

        n_elements = len(self.symbols)
        self.pair_coeffs = nn.Parameter(
            torch.zeros(
                n_elements,
                n_elements,
                self.config.n_pair_basis,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.config.optimize_pair,
        )
        self.density_coeffs = nn.Parameter(
            torch.zeros(
                n_elements,
                n_elements,
                self.config.n_density_basis,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.config.optimize_density,
        )

        pair_mask_tensor = _correction_mask(
            pair_mask,
            n_elements,
            symmetric=True,
            dtype=self.dtype,
            device=self.device,
        )
        density_mask_tensor = _correction_mask(
            density_mask,
            n_elements,
            symmetric=False,
            dtype=self.dtype,
            device=self.device,
        )
        self.register_buffer("_pair_mask", pair_mask_tensor, persistent=True)
        self.register_buffer("_density_mask", density_mask_tensor, persistent=True)

    def pair_correction(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate ``phi`` corrections for every ordered pair channel."""

        basis = _chebyshev_basis(
            r, self.config.n_pair_basis, self.r_min, self.r_max
        )
        envelope = _smooth_radial_window(r, self.r_min, self.r_max)
        coeffs = 0.5 * (self.pair_coeffs + self.pair_coeffs.transpose(0, 1))
        coeffs = coeffs * self._pair_mask[..., None]
        return (
            self.config.pair_scale
            * envelope[..., None, None]
            * torch.einsum("...k,ijk->...ij", basis, coeffs)
        )

    def density_correction(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate directed density corrections for every channel."""

        basis = _chebyshev_basis(
            r, self.config.n_density_basis, self.r_min, self.r_max
        )
        envelope = _smooth_radial_window(r, self.r_min, self.r_max)
        coeffs = self.density_coeffs * self._density_mask[..., None]
        return (
            self.config.density_scale
            * envelope[..., None, None]
            * torch.einsum("...k,ijk->...ij", basis, coeffs)
        )

    def energy(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        types: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        vectors = positions[j] + shifts @ cell - positions[i]
        distances = torch.linalg.norm(vectors, dim=1)

        density_values = self._evaluate_r_table(
            self._density[types[i], types[j]], distances
        )
        density_values = density_values + self.density_correction(distances)[
            torch.arange(distances.numel(), device=self.device),
            types[i],
            types[j],
        ]
        atomic_density = torch.zeros(
            positions.shape[0], dtype=self.dtype, device=self.device
        ).index_add(0, i, density_values)
        embedding_energy = self._evaluate_rho_table(
            self._embedding[types], atomic_density
        ).sum()

        rphi_values = self._evaluate_r_table(
            self._rphi[types[i], types[j]], distances
        )
        safe_distances = distances.clamp_min(torch.finfo(self.dtype).eps)
        pair_values = rphi_values / safe_distances
        pair_values = pair_values + self.pair_correction(distances)[
            torch.arange(distances.numel(), device=self.device),
            types[i],
            types[j],
        ]
        pair_energy = 0.5 * pair_values.sum()
        return embedding_energy + pair_energy

    def correction_dict(self) -> dict[str, dict[str, list[float]]]:
        """Return correction coefficients keyed by channel name."""

        pair_coeffs = 0.5 * (self.pair_coeffs + self.pair_coeffs.transpose(0, 1))
        return {
            "pair": {
                self._channel_name(i, j): pair_coeffs[i, j].detach().cpu().tolist()
                for i in range(len(self.symbols))
                for j in range(i + 1)
            },
            "density": {
                self._directed_channel_name(i, j): self.density_coeffs[i, j]
                .detach()
                .cpu()
                .tolist()
                for i in range(len(self.symbols))
                for j in range(len(self.symbols))
            },
        }

    def corrected_density_table(self) -> torch.Tensor:
        """Return the corrected raw density table.

        The returned tensor has shape ``(target, source, r_grid)`` and is in the
        same convention as :class:`EAMFSData.density`.
        """

        r_grid = torch.as_tensor(
            self.potential.r_grid,
            dtype=self.dtype,
            device=self.device,
        )
        correction = self.density_correction(r_grid).movedim(0, -1)
        density = torch.as_tensor(
            self.potential.density,
            dtype=self.dtype,
            device=self.device,
        )
        return density + correction

    def corrected_rphi_table(self) -> torch.Tensor:
        """Return the corrected raw ``r * phi`` table.

        The pair spline correction is defined for ``phi(r)``.  The EAM/FS file
        stores ``r * phi(r)``, so the correction is multiplied by the radial grid
        before being added to the baseline table.
        """

        r_grid = torch.as_tensor(
            self.potential.r_grid,
            dtype=self.dtype,
            device=self.device,
        )
        correction_phi = self.pair_correction(r_grid).movedim(0, -1)
        rphi = torch.as_tensor(
            self.potential.rphi,
            dtype=self.dtype,
            device=self.device,
        )
        correction_rphi = correction_phi * r_grid[None, None, :]
        corrected = rphi + correction_rphi
        return 0.5 * (corrected + corrected.transpose(0, 1))

    def to_eam_fs_potential(self) -> EAMFSData:
        """Materialize the current corrected model as an ``EAMFSData``."""

        return EAMFSData(
            comments=self.potential.comments,
            symbols=self.potential.symbols,
            atomic_numbers=np.array(self.potential.atomic_numbers, copy=True),
            masses=np.array(self.potential.masses, copy=True),
            lattice_constants=np.array(self.potential.lattice_constants, copy=True),
            lattice_types=self.potential.lattice_types,
            nrho=self.potential.nrho,
            drho=self.potential.drho,
            nr=self.potential.nr,
            dr=self.potential.dr,
            cutoff=self.potential.cutoff,
            embedding=np.array(self.potential.embedding, copy=True),
            density=self.corrected_density_table().detach().cpu().numpy(),
            rphi=self.corrected_rphi_table().detach().cpu().numpy(),
        )

    def _channel_name(self, i: int, j: int) -> str:
        return f"{self.symbols[i]}-{self.symbols[j]}"

    def _directed_channel_name(self, target: int, source: int) -> str:
        return f"{self.symbols[target]}|{self.symbols[source]}"


def _lammps_eam_spline_coefficients(values: np.ndarray) -> np.ndarray:
    """Build the cubic Hermite coefficients used by LAMMPS EAM.

    The last axis of ``values`` is the tabulated grid.  The returned last axis
    stores ``a, b, c, d`` for ``((a*p + b)*p + c)*p + d`` on each grid interval.
    """

    values = np.asarray(values, dtype=float)
    n_grid = values.shape[-1]
    if n_grid < 4:
        raise ValueError("LAMMPS EAM interpolation requires at least 4 grid points")

    slopes = np.empty_like(values)
    slopes[..., 0] = values[..., 1] - values[..., 0]
    slopes[..., 1] = 0.5 * (values[..., 2] - values[..., 0])
    slopes[..., -2] = 0.5 * (values[..., -1] - values[..., -3])
    slopes[..., -1] = values[..., -1] - values[..., -2]
    slopes[..., 2:-2] = (
        (values[..., :-4] - values[..., 4:])
        + 8.0 * (values[..., 3:-1] - values[..., 1:-3])
    ) / 12.0
    # 5 point center finite difference derivatives

    delta_y = values[..., 1:] - values[..., :-1]
    a = slopes[..., :-1] + slopes[..., 1:] - 2.0 * delta_y
    b = 3.0 * delta_y - 2.0 * slopes[..., :-1] - slopes[..., 1:]
    c = slopes[..., :-1]
    d = values[..., :-1]
    return np.stack((a, b, c, d), axis=-1)


def _evaluate_lammps_eam_spline(
    coeffs: torch.Tensor,
    x: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Evaluate LAMMPS EAM's per-interval cubic polynomial."""

    n_intervals = coeffs.shape[-2]
    scaled = (x / step).clamp(0.0, float(n_intervals))
    lower = torch.floor(scaled).to(torch.long).clamp(min=0, max=n_intervals - 1)
    fraction = (scaled - lower.to(scaled.dtype)).clamp(max=1.0)
    selected = torch.gather(
        coeffs,
        -2,
        lower.unsqueeze(-1).unsqueeze(-1).expand(*lower.shape, 1, 4),
    ).squeeze(-2)
    return (
        ((selected[..., 0] * fraction + selected[..., 1]) * fraction + selected[..., 2])
        * fraction
        + selected[..., 3]
    )


def _correction_mask(
    mask: torch.Tensor | np.ndarray | None,
    n_elements: int,
    *,
    symmetric: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        result = torch.ones(n_elements, n_elements, dtype=dtype, device=device)
    else:
        result = torch.as_tensor(mask, dtype=dtype, device=device)
        if result.shape != (n_elements, n_elements):
            raise ValueError(
                f"Correction mask must have shape {(n_elements, n_elements)}, "
                f"got {tuple(result.shape)}."
            )
    if symmetric:
        result = torch.maximum(result, result.T)
    return result


def _chebyshev_basis(
    r: torch.Tensor,
    n_basis: int,
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    if n_basis < 1:
        raise ValueError("n_basis must be at least 1 for Chebyshev corrections.")

    r_flat = r.reshape(-1)
    x = 2.0 * (r_flat - r_min) / (r_max - r_min) - 1.0
    x = x.clamp(-1.0, 1.0)
    basis_terms = [torch.ones_like(x)]
    if n_basis > 1:
        basis_terms.append(x)
    for order in range(2, n_basis):
        basis_terms.append(2.0 * x * basis_terms[order - 1] - basis_terms[order - 2])
    basis = torch.stack(basis_terms, dim=-1)
    return basis.reshape(*r.shape, n_basis)


def _smooth_radial_window(
    r: torch.Tensor,
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    scaled = ((r - r_min) / (r_max - r_min)).clamp(0.0, 1.0)
    return torch.sin(torch.pi * scaled).square()
