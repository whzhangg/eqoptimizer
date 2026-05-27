import dataclasses
import pathlib
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .solver import PycalphadPhaseSolver
from ..dtype import TORCH_FLOAT
from ..utilities import PRESSURE, as_float_tensor, multi_simplex_samples_dirichlet


def default_pycalphad_components(database) -> tuple[str, ...]:
    """Return components passed to pycalphad.

    Keep VA because TDB phase models often require it, but drop /- because it
    is charge/electron bookkeeping rather than a composition axis here.
    """
    return tuple(
        component
        for component in sorted(str(component) for component in database.elements)
        if component != "/-"
    )


def real_elements_from_components(components: Sequence[str]) -> tuple[str, ...]:
    """Return real composition axes used by observed data and torch models."""
    return tuple(component for component in components if component not in {"VA", "/-"})


@dataclasses.dataclass(frozen=True)
class PycalphadEvaluation:
    """Fixed, non-differentiable pycalphad phase evaluations."""

    phase_name: str
    elements: tuple[str, ...]
    site_fraction_symbols: tuple[str, ...]
    y: Tensor
    x: Tensor
    gibbs_energy: Tensor
    converged: tuple[bool, ...] | None = None
    residual_norm: Tensor | None = None


class PycalphadReferenceModel:
    """Non-differentiable interface to a pycalphad phase model.

    This class intentionally uses pycalphad for reference Gibbs-energy
    evaluation and returns detached torch tensors.
    """

    def __init__(
        self,
        database,
        phase_name: str,
        components: Sequence[str] | None = None,
        *,
        pressure: float = PRESSURE,
        device=None,
    ) -> None:
        from pycalphad import Database, Model

        if isinstance(database, (str, pathlib.Path)):
            database = Database(str(database))

        self.database = database
        self.phase_name = phase_name
        self.components = tuple(components or default_pycalphad_components(database))
        self.elements = real_elements_from_components(self.components)
        self.pressure = pressure
        self.dtype = TORCH_FLOAT
        self.device = device
        self.pycalphad_model = Model(database, self.components, phase_name)
        self.phase_solver = PycalphadPhaseSolver(
            database,
            phase_name,
            self.components,
            pressure=self.pressure,
        )
        #self.phase_record = database.phases[phase_name]
        self.site_fraction_symbols = tuple(
            self._site_fraction_symbol(site_fraction)
            for site_fraction in self.pycalphad_model.site_fractions
        ) # eg: ('FCC0AL', 'FCC0ZN')
        self.site_fraction_elements = tuple(
            str(site_fraction.species)
            for site_fraction in self.pycalphad_model.site_fractions
        ) # eg: ('AL', 'ZN')


    @classmethod
    def from_tdb_file(
        cls,
        tdbfilename: str | pathlib.Path,
        phase_name: str,
        components: Sequence[str] | None = None,
        **kwargs,
    ) -> "PycalphadReferenceModel":
        return cls(tdbfilename, phase_name, components, **kwargs)


    @staticmethod
    def _site_fraction_symbol(site_fraction) -> str:
        return (
            f"{site_fraction.phase_name}"
            f"{site_fraction.sublattice_index}"
            f"{site_fraction.species}"
        )

    def _to_tensor(self, value) -> Tensor:
        return torch.as_tensor(
            np.array(value, copy=True),
            dtype=self.dtype,
            device=self.device,
        ).detach()

    def _calculate_points(
        self,
        points,
        temperature: float,
        *,
        output: str = "GM",
    ) -> PycalphadEvaluation:
        from pycalphad import calculate

        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]

        result = calculate(
            self.database,
            self.components,
            self.phase_name,
            output=output,
            T=float(temperature),
            P=float(self.pressure),
            points=points,
        )

        elements = tuple(str(component) for component in result.component.values)
        gibbs_energy = np.asarray(result[output].values, dtype=float).reshape(-1)
        x = np.asarray(result.X.values, dtype=float).reshape(-1, len(elements))
        y = np.asarray(result.Y.values, dtype=float).reshape(
            -1, len(self.site_fraction_symbols)
        )

        return PycalphadEvaluation(
            phase_name=self.phase_name,
            elements=elements,
            site_fraction_symbols=self.site_fraction_symbols,
            y=self._to_tensor(y),
            x=self._to_tensor(x),
            gibbs_energy=self._to_tensor(gibbs_energy),
        )

    def _evaluation_from_solver_results(self, results) -> PycalphadEvaluation:
        return PycalphadEvaluation(
            phase_name=self.phase_name,
            elements=self.elements,
            site_fraction_symbols=self.site_fraction_symbols,
            y=self._to_tensor([result.y for result in results]),
            x=self._to_tensor([result.x for result in results]),
            gibbs_energy=self._to_tensor([result.gibbs_energy for result in results]),
            converged=tuple(result.converged for result in results),
            residual_norm=self._to_tensor([result.residual_norm for result in results]),
        )

    def gibbs_energy(
        self,
        compositions,
        temperature: float,
    ) -> PycalphadEvaluation:
        """Evaluate G at imposed component compositions.

        For a one-sublattice substitutional phase, component composition and
        site fractions are identical. Multi-sublattice phases are evaluated by
        solving the constrained one-phase minimization problem.
        """
        compositions = np.asarray(compositions, dtype=float)
        if compositions.ndim == 1:
            compositions = compositions[None, :]
        if compositions.shape[-1] != len(self.elements):
            raise ValueError(
                f"Expected {len(self.elements)} composition columns "
                f"for {self.elements}, got {compositions.shape[-1]}."
            )

        if len(self.pycalphad_model.site_ratios) != 1:
            results = [
                self.phase_solver.solve(composition, temperature)
                for composition in compositions
            ]
            return self._evaluation_from_solver_results(results)

        element_column = {
            element: compositions[:, index]
            for index, element in enumerate(self.elements)
        }
        points = np.stack(
            [element_column[element] for element in self.site_fraction_elements],
            axis=-1,
        )
        return self._calculate_points(points, temperature)

    def sample_internal_dof(
        self,
        n_samples_each_side: int,
    ) -> np.ndarray:
        """Sample pycalphad internal site fractions for this phase."""
        sublattice_to_indices: dict[int, list[int]] = {}
        for index, site_fraction in enumerate(self.pycalphad_model.site_fractions):
            sublattice_to_indices.setdefault(
                int(site_fraction.sublattice_index), []
            ).append(index)

        sublattice_indices = list(sublattice_to_indices.values())
        sampled = multi_simplex_samples_dirichlet(
            [len(indices) for indices in sublattice_indices],
            n_samples_each_side=n_samples_each_side,
        ).detach().cpu().numpy()

        points = np.empty(
            (sampled.shape[0], len(self.site_fraction_symbols)),
            dtype=float,
        )
        column_start = 0
        for indices in sublattice_indices:
            column_stop = column_start + len(indices)
            points[:, indices] = sampled[:, column_start:column_stop]
            column_start = column_stop
        return points

    def sampled_internal_dof(
        self,
        temperature: float,
        *,
        n_samples_each_side: int = 16,
    ) -> PycalphadEvaluation:
        """Sample y, then return fixed pycalphad G(y), X(y), and Y."""
        points = self.sample_internal_dof(n_samples_each_side)
        return self._calculate_points(points, temperature)


def demo() -> None:
    from pycalphad import Database
    from ..models import CorrectedGibbsModel, SolidSolutionModel

    tdb_path = pathlib.Path(__file__).resolve().parents[3] / "examples" / "CPDDB_AlZn.tdb"
    db = Database(str(tdb_path))
    reference = PycalphadReferenceModel(
        db,
        "FCC",
    )

    fixed = reference.gibbs_energy([[0.25, 0.75], [0.80, 0.20]], 600.0)
    print(fixed)
    sampled = reference.sampled_internal_dof(600.0, n_samples_each_side=5)
    correction = SolidSolutionModel(
        2,
        polynomial_order=1,
        interaction_order=1,
        name="FCC_CORRECTION",
        elements=reference.elements,
    )
    corrected = CorrectedGibbsModel(reference, correction)
    phi = corrected.grand_potential(
        as_float_tensor([0.0, 0.0]),
        600.0,
        samples=sampled,
    )
    phi.backward()

    print(f"phase: {reference.phase_name}")
    print(f"pycalphad components: {reference.components}")
    print(f"elements: {reference.elements}")
    print(f"G at imposed X: {fixed.gibbs_energy.tolist()}")
    print(f"sampled X: {sampled.x.tolist()}")
    print(f"sampled G0: {sampled.gibbs_energy.tolist()}")
    print(f"Phi with correction: {float(phi.detach()):.6f}")
    print(
        "correction grad norm: "
        f"{float(sum(p.grad.abs().sum() for p in correction.parameters())):.6f}"
    )
    print(f"pycalphad G0 requires_grad: {sampled.gibbs_energy.requires_grad}")

