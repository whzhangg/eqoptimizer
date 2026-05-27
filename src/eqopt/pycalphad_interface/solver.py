r"""
Omit the dependence of $G_m$ on $T$ and $P$, we minimize $y$ with the constraints on $M$:
$$
L = G^{\alpha}(\mathbf{y}) + \sum_A (\mathbb{M}_A^{\alpha} - M_A^{\alpha})\cdot \mu_A^{\alpha}
$$
we have:
$$
\begin{gather}
\frac{\partial L}{\partial y_i^{\alpha}} = \frac{\partial G^{\alpha}}{\partial y_i^{\alpha}} - \sum_A \frac{\partial M_A^{\alpha}}{\partial y_i^{\alpha}} \mu_A = 0 ;\quad\quad
\frac{\partial L}{\partial \mu_A} = \mathbb{M}_A^{\alpha} - M_A^{\alpha} = 0
\end{gather}
$$
Using the Newton's method, we have one following equation for each internal coordinates:
$$
\frac{\partial G^{\alpha}}{\partial y_i^{\alpha}} - \sum_A \frac{\partial M_A^{\alpha}}{\partial y_i^{\alpha}} \mu_A + \sum_{j}\frac{\partial^2 G}{\partial y_i \partial y_j}\Delta y_j - \sum_A \frac{\partial M_A^{\alpha}}{\partial y_i^{\alpha}} \Delta \mu_A= 0
$$
From the second equation, one for each component, if we choose initial condition so that $M_{A} = \mathbb{M}_A$, then we simply have:
$$
\sum_i \frac{\partial M_A}{\partial y_i} \Delta y_i = 0
$$
The first equation can be rewritten by $\mu_A'=\mu_A + \Delta \mu_A$, and where $\mathbf{H}$ is the Hessian:
$$
\begin{gather}
\sum_{j}\frac{\partial^2 G}{\partial y_i \partial y_j}\Delta y_j = \sum_A \frac{\partial M_A^{\alpha}}{\partial y_i^{\alpha}} \mu_A' - \frac{\partial G^{\alpha}}{\partial y_i^{\alpha}} \\
\Delta y_i = \sum_A \sum_j [\mathbf{H}^{-1}]_{ij} \frac{\partial M_A^{\alpha}}{\partial y_j^{\alpha}} \mu_A' - \sum_j [\mathbf{H}^{-1}]_{ij}  \frac{\partial G^{\alpha}}{\partial y_j^{\alpha}}
\end{gather}
$$
we substitute $\Delta y_i$ into $\sum_i \frac{\partial M_A}{\partial y_i} \Delta y_i = 0$ to arrive at a set of equations, one for each component, that allow us to solve $\mu_A'$:
$$
\sum_i \frac{\partial M_A}{\partial y_i} \left[\sum_A \sum_j [\mathbf{H}^{-1}]_{ij} \frac{\partial M_A^{\alpha}}{\partial y_j^{\alpha}} \mu_A' - \sum_j [\mathbf{H}^{-1}]_{ij}  \frac{\partial G^{\alpha}}{\partial y_j^{\alpha}}\right] = 0
$$
From which $\mu_A'$ can be solved directly, and then, $\Delta \mathrm{y}$ can be solved.

To keep $0\leq y\leq 1$, we can use $y = \mathrm{sigmoid}(z)$ and optimize instead on $z$
"""

from __future__ import annotations

import dataclasses
import pathlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np

from ..utilities import PRESSURE, multi_simplex_samples_dirichlet


@dataclasses.dataclass(frozen=True)
class ConstrainedPhaseResult:
    phase_name: str
    elements: tuple[str, ...]
    site_fraction_names: tuple[str, ...]
    target_x: np.ndarray
    x: np.ndarray
    y: np.ndarray
    gibbs_energy: float
    chemical_potentials: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float


def default_pycalphad_components(database) -> tuple[str, ...]:
    return tuple(
        component
        for component in sorted(str(component) for component in database.elements)
        if component != "/-"
    )


@dataclasses.dataclass(frozen=True)
class NewtonLagrangeResult:
    variables: np.ndarray
    multipliers: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float


class LagrangeProblem(ABC):
    """Interface consumed by the Newton solver.

    The solver does not know what the variables mean. It only sees a Lagrangian
    L(y, lambda) = f(y) + lambda dot c(y) through derivatives.
    """

    @property
    @abstractmethod
    def n_variables(self) -> int:
        pass

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        pass

    @abstractmethod
    def value(self, variables: np.ndarray, multipliers: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient_variables(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_multipliers(self, variables: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hessian_variables(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def constraint_jacobian(self, variables: np.ndarray) -> np.ndarray:
        pass

    def project_variables(self, variables: np.ndarray) -> np.ndarray:
        return np.asarray(variables, dtype=float)

    def in_domain(self, variables: np.ndarray) -> bool:
        return np.all(np.isfinite(variables))

    def residual(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([
            self.gradient_variables(variables, multipliers),
            self.gradient_multipliers(variables),
        ])

    def jacobian(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
        regularization: float = 0.0,
    ) -> np.ndarray:
        n_y = self.n_variables
        n_c = self.n_constraints
        constraint_jac = self.constraint_jacobian(variables)
        return np.block([
            [
                self.hessian_variables(variables, multipliers)
                + regularization * np.eye(n_y),
                constraint_jac.T,
            ],
            [constraint_jac, np.zeros((n_c, n_c))],
        ])


class NewtonLagrangeSolver:
    """Newton solver for a LagrangeProblem."""

    def __init__(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1.0e-9,
        damping: float = 1.0,
        regularization: float = 1.0e-10,
        max_line_search: int = 20,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.regularization = regularization
        self.max_line_search = max_line_search

    def solve(
        self,
        problem: LagrangeProblem,
        initial_variables: Sequence[float],
        initial_multipliers: Sequence[float] | None = None,
    ) -> NewtonLagrangeResult:
        variables = problem.project_variables(np.asarray(initial_variables, dtype=float))
        if initial_multipliers is None:
            multipliers = np.zeros(problem.n_constraints)
        else:
            multipliers = np.asarray(initial_multipliers, dtype=float).copy()

        converged = False
        residual_norm = np.inf
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            residual = problem.residual(variables, multipliers)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm < self.tol:
                converged = True
                break

            jacobian = problem.jacobian(
                variables,
                multipliers,
                regularization=self.regularization,
            )
            rhs = -residual
            try:
                step = np.linalg.solve(jacobian, rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, rhs, rcond=None)[0]

            dy = step[: problem.n_variables]
            dlambda = step[problem.n_variables :]
            accepted = False
            step_scale = self.damping
            for _ in range(self.max_line_search):
                candidate_variables = problem.project_variables(
                    variables + step_scale * dy
                )
                if problem.in_domain(candidate_variables):
                    candidate_multipliers = multipliers + step_scale * dlambda
                    candidate_norm = float(
                        np.linalg.norm(
                            problem.residual(
                                candidate_variables,
                                candidate_multipliers,
                            )
                        )
                    )
                    if candidate_norm < residual_norm:
                        variables = candidate_variables
                        multipliers = candidate_multipliers
                        residual_norm = candidate_norm
                        accepted = True
                        break
                step_scale *= 0.5

            if not accepted:
                variables = problem.project_variables(variables + step_scale * dy)
                multipliers = multipliers + step_scale * dlambda

        residual_norm = float(np.linalg.norm(problem.residual(variables, multipliers)))
        converged = converged or residual_norm < self.tol
        return NewtonLagrangeResult(
            variables=variables,
            multipliers=multipliers,
            converged=converged,
            iterations=iteration,
            residual_norm=residual_norm,
        )


class PycalphadCompositionLagrangeProblem(LagrangeProblem):
    """Lagrange problem for min_y G(y) subject to X(y)=target."""

    def __init__(
        self,
        phase_solver: PycalphadPhaseSolver,
        target_x: np.ndarray,
        temperature: float,
        *,
        min_site_fraction: float = 1.0e-10,
    ) -> None:
        self.phase_solver = phase_solver
        self.target_x = np.asarray(target_x, dtype=float)
        self.temperature = float(temperature)
        self.min_site_fraction = min_site_fraction

        self._n_constraints = (
            max(len(phase_solver.elements) - 1, 0)
            + len(phase_solver._sublattice_indices)
        )

    @property
    def n_variables(self) -> int:
        return len(self.phase_solver.site_fractions)

    @property
    def n_constraints(self) -> int:
        return self._n_constraints

    def value(self, variables: np.ndarray, multipliers: np.ndarray) -> float:
        return (
            self.phase_solver.gibbs_energy(variables, self.temperature)
            + float(multipliers @ self.gradient_multipliers(variables))
        )

    def gradient_variables(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        constraints, constraint_jac, _ = self._constraints(variables)
        del constraints
        return (
            self.phase_solver.gibbs_gradient(variables, self.temperature)
            + constraint_jac.T @ multipliers
        )

    def gradient_multipliers(self, variables: np.ndarray) -> np.ndarray:
        constraints, _, _ = self._constraints(variables)
        return constraints

    def hessian_variables(
        self,
        variables: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        _, _, constraint_hessians = self._constraints(variables)
        return (
            self.phase_solver.gibbs_hessian(variables, self.temperature)
            + np.tensordot(multipliers, constraint_hessians, axes=(0, 0))
        )

    def constraint_jacobian(self, variables: np.ndarray) -> np.ndarray:
        _, jacobian, _ = self._constraints(variables)
        return jacobian

    def project_variables(self, variables: np.ndarray) -> np.ndarray:
        return self.phase_solver._normalize_sublattices(
            np.clip(np.asarray(variables, dtype=float), self.min_site_fraction, None)
        )

    def in_domain(self, variables: np.ndarray) -> bool:
        return (
            np.all(np.isfinite(variables))
            and np.all(variables > self.min_site_fraction)
        )

    def _constraints(
        self,
        variables: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.phase_solver.constraints(
            variables,
            self.target_x,
            self.temperature,
        )


class PycalphadPhaseSolver:
    """Solve one-phase constrained minimization using pycalphad expressions.

    The public surface is intentionally small. pycalphad internals are used only
    to compile Gibbs energy, composition, and their derivatives with respect to
    site fractions.
    """

    def __init__(
        self,
        database,
        phase_name: str,
        components: Sequence[str] | None = None,
        *,
        pressure: float = PRESSURE,
    ) -> None:
        from pycalphad import Database, Model
        from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
        from pycalphad.codegen.sympydiff_utils import build_functions
        from pycalphad.core.utils import unpack_species
        from pycalphad import variables as v

        if isinstance(database, (str, pathlib.Path)):
            database = Database(str(database))

        self.database = database
        self.phase_name = phase_name
        self.components = tuple(components or default_pycalphad_components(database))
        self.pressure = float(pressure)

        species_components = sorted(unpack_species(database, self.components))
        self.model = Model(database, species_components, phase_name)
        self.models = {phase_name: self.model}

        state_conditions = OrderedDict([(v.T, 1.0)])
        self.phase_record_factory = PhaseRecordFactory(
            database,
            species_components,
            state_conditions,
            self.models,
        )
        self.state_variables = tuple(self.phase_record_factory.state_variables)
        self.site_fractions = tuple(self.model.site_fractions)
        self.variables = tuple(self.state_variables + tuple(self.model.site_fractions))
        self.elements = tuple(self.phase_record_factory.nonvacant_elements)

        # used to get Gibbs energy and its derivatives
        self._gibbs = build_functions(
            self.model.GM,
            self.variables,
            wrt=self.site_fractions,
            include_obj=True,
            include_grad=True,
            include_hess=True,
        )

        # used to get composition and its derivatives
        moles = [
            self.model.moles(element, per_formula_unit=True)
            for element in self.elements
        ]
        total_moles = sum(moles)
        self._x_funcs = [
            build_functions(
                mole / total_moles,
                self.variables,
                wrt=self.site_fractions,
                include_obj=True,
                include_grad=True,
                include_hess=True,
            )
            for mole in moles
        ]

        self._sublattice_indices = self._build_sublattice_indices()

    def _build_sublattice_indices(self) -> list[list[int]]:
        sublattice_indices: dict[int, list[int]] = {}
        for index, site_fraction in enumerate(self.site_fractions):
            sublattice_indices.setdefault(
                int(site_fraction.sublattice_index),
                [],
            ).append(index)
        return list(sublattice_indices.values())

    def _args(self, y: np.ndarray, temperature: float) -> tuple[float, ...]:
        state_values = []
        for variable in self.state_variables:
            name = str(variable)
            if name == "T":
                state_values.append(float(temperature))
            elif name == "P":
                state_values.append(self.pressure)
            elif name == "N":
                state_values.append(1.0)
            else:
                raise ValueError(f"Unsupported pycalphad state variable {variable!r}.")
        return tuple(state_values + list(np.asarray(y, dtype=float)))

    @staticmethod
    def _as_array(value) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def gibbs_energy(self, y: Sequence[float], temperature: float) -> float:
        return float(self._gibbs.func(*self._args(np.asarray(y), temperature)))

    def gibbs_gradient(self, y: Sequence[float], temperature: float) -> np.ndarray:
        return self._as_array(self._gibbs.grad(*self._args(np.asarray(y), temperature)))

    def gibbs_hessian(self, y: Sequence[float], temperature: float) -> np.ndarray:
        return self._as_array(self._gibbs.hess(*self._args(np.asarray(y), temperature)))

    def composition(self, y: Sequence[float], temperature: float) -> np.ndarray:
        args = self._args(np.asarray(y), temperature)
        return np.array([float(func.func(*args)) for func in self._x_funcs])

    def composition_jacobian(self, y: Sequence[float], temperature: float) -> np.ndarray:
        args = self._args(np.asarray(y), temperature)
        return np.vstack([self._as_array(func.grad(*args)) for func in self._x_funcs])

    def composition_hessians(self, y: Sequence[float], temperature: float) -> np.ndarray:
        args = self._args(np.asarray(y), temperature)
        return np.stack([self._as_array(func.hess(*args)) for func in self._x_funcs])

    def constraints(
        self,
        y: np.ndarray,
        target_x: np.ndarray,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return c(y), dc/dy and d2c/dy2 for independent constraints."""
        x = self.composition(y, temperature)
        x_jac = self.composition_jacobian(y, temperature)
        x_hess = self.composition_hessians(y, temperature)

        # Composition mole fractions sum to one, so the last composition
        # equation is redundant.
        c_parts = [x[:-1] - target_x[:-1]]
        jac_parts = [x_jac[:-1]]
        hess_parts = [x_hess[:-1]]

        n_y = len(y)
        for indices in self._sublattice_indices:
            row = np.zeros(n_y)
            row[indices] = 1.0
            hess = np.zeros((n_y, n_y))
            c_parts.append(np.array([np.sum(y[indices]) - 1.0]))
            jac_parts.append(row[None, :])
            hess_parts.append(hess[None, :, :])

        return (
            np.concatenate(c_parts),
            np.vstack(jac_parts),
            np.concatenate(hess_parts, axis=0),
        )

    def sample_internal_dof(self, n_samples_each_side: int = 16) -> np.ndarray:
        sampled = multi_simplex_samples_dirichlet(
            [len(indices) for indices in self._sublattice_indices],
            n_samples_each_side=n_samples_each_side,
        ).detach().cpu().numpy()

        points = np.empty((sampled.shape[0], len(self.site_fractions)), dtype=float)
        column_start = 0
        for indices in self._sublattice_indices:
            column_stop = column_start + len(indices)
            points[:, indices] = sampled[:, column_start:column_stop]
            column_start = column_stop
        return points

    def initial_y_from_samples(
        self,
        target_x: np.ndarray,
        temperature: float,
        *,
        n_samples_each_side: int = 24,
    ) -> np.ndarray:
        samples = self.sample_internal_dof(n_samples_each_side)
        compositions = np.array([
            self.composition(sample, temperature)
            for sample in samples
        ])
        errors = np.linalg.norm(compositions - target_x[None, :], axis=1)
        return samples[int(np.argmin(errors))].copy()

    def solve(
        self,
        target_x: Mapping[str, float] | Sequence[float],
        temperature: float,
        *,
        initial_y: Sequence[float] | None = None,
        n_samples_each_side: int = 24,
        max_iter: int = 50,
        tol: float = 1.0e-9,
        damping: float = 1.0,
        min_site_fraction: float = 1.0e-10,
        regularization: float = 1.0e-10,
    ) -> ConstrainedPhaseResult:
        target = self._target_array(target_x)
        if initial_y is None:
            initial_variables = self.initial_y_from_samples(
                target,
                temperature,
                n_samples_each_side=n_samples_each_side,
            )
        else:
            initial_variables = np.asarray(initial_y, dtype=float).copy()

        problem = PycalphadCompositionLagrangeProblem(
            self,
            target,
            temperature,
            min_site_fraction=min_site_fraction,
        )
        newton_result = NewtonLagrangeSolver(
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            regularization=regularization,
        ).solve(problem, initial_variables)

        y = newton_result.variables
        multipliers = newton_result.multipliers
        x = self.composition(y, temperature)
        gibbs = self.gibbs_energy(y, temperature)

        return ConstrainedPhaseResult(
            phase_name=self.phase_name,
            elements=self.elements,
            site_fraction_names=tuple(str(site_fraction) for site_fraction in self.site_fractions),
            target_x=target,
            x=x,
            y=y,
            gibbs_energy=gibbs,
            chemical_potentials=-multipliers[: max(len(self.elements) - 1, 0)],
            converged=newton_result.converged,
            iterations=newton_result.iterations,
            residual_norm=newton_result.residual_norm,
        )

    def _target_array(self, target_x: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        if isinstance(target_x, Mapping):
            target = np.array([target_x[element] for element in self.elements], dtype=float)
        else:
            target = np.asarray(target_x, dtype=float)
        if target.shape != (len(self.elements),):
            raise ValueError(
                f"Expected target composition for {len(self.elements)} elements "
                f"{self.elements}, got shape {target.shape}."
            )
        target = target / target.sum()
        return target

    def _normalize_sublattices(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).copy()
        for indices in self._sublattice_indices:
            total = np.sum(y[indices])
            if total <= 0:
                y[indices] = 1.0 / len(indices)
            else:
                y[indices] = y[indices] / total
        return y


if __name__ == "__main__":
    db_path = pathlib.Path(__file__).resolve().parents[3] / "examples" / "CPDDB_WRe.tdb"
    solver = PycalphadPhaseSolver(db_path, "SIGMA")
    result = solver.solve({"W": 0.5, "RE": 0.5}, 2000.0)
    print(result)
