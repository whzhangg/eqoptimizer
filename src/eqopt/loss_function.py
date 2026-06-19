import itertools
import math
import torch
import torch.nn.functional as F
from collections.abc import Mapping, Sequence, Set
import dataclasses

from .models.system_abc import ThermodynamicSystem
from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .utilities import R
from .phase import PhaseID, PhaseEquilibrium


def _format_scalar_for_print(value: torch.Tensor | float) -> str:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().reshape(()))
    return f"{value:11.3e}"


def _composition_tensor(
    composition: Mapping[str, float | torch.Tensor],
    elements: Sequence[str],
) -> torch.Tensor:
    values = torch.as_tensor(
        [composition.get(element, 0.0) for element in elements],
        device=DEFAULT_DEVICE,
        dtype=DEFAULT_TYPE,
    )
    return values / values.sum().clamp_min(1.0e-12)


def _matrix_rank(matrix: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(matrix.detach()).item())


def _condition_number(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.detach())
    if singular_values.numel() == 0:
        return float("inf")
    min_singular = float(singular_values[-1].cpu())
    if min_singular <= 0.0:
        return float("inf")
    return float((singular_values[0] / singular_values[-1]).cpu())


@dataclasses.dataclass
class EquilibriumLossRecord:
    equilibrium: PhaseEquilibrium
    mu_strategy: str
    mu: Mapping[str, torch.Tensor]
    phi: Mapping[PhaseID, torch.Tensor]
    stable_phase_ids: Set[PhaseID]
    stable_loss: torch.Tensor
    unstable_loss: torch.Tensor


class PhaseEquilibriumOptState(torch.nn.Module):
    analytic_condition_threshold: float = 1.0e10
    def __init__(self,
        equilibrium: PhaseEquilibrium,
        *,
        mu_strategy: str = "auto",
    ):
        super().__init__()
        if len(equilibrium.phases) != len(equilibrium.phase_compositions):
            raise ValueError(
                "PhaseEquilibrium.phases and phase_compositions must have "
                "the same length."
            )
        if not equilibrium.phases:
            raise ValueError("PhaseEquilibrium must contain at least one phase.")

        self.equilibrium = equilibrium
        self.elements = tuple(sorted(equilibrium.chemical_system))
        self.stable_phase_ids = set(self.equilibrium.phases)

        self.x_matrix = torch.stack(
            [_composition_tensor(composition, self.elements) 
            for composition in self.equilibrium.phase_compositions], dim=0
        )
        self.x_rank = _matrix_rank(self.x_matrix)
        self.strategy = self._select_mu_strategy(mu_strategy)
        if self.strategy == "latent":
            self._prepare_augmented_mu_system()
            self.mu = torch.nn.Parameter(
                torch.empty(
                    len(self.gauge_component_indices),
                    device=DEFAULT_DEVICE,
                    dtype=DEFAULT_TYPE,
                )
            )
            torch.nn.init.zeros_(self.mu)
        else:
            self.register_parameter("mu", None)


    def _select_mu_strategy(self, mu_strategy: str) -> str:
        if mu_strategy not in ("auto", "analytic", "latent"):
            raise ValueError(
                "mu_strategy must be one of 'auto', 'analytic', or 'latent'."
            )
        if mu_strategy != "auto":
            return mu_strategy

        if self.x_rank < len(self.elements):
            return "latent"
        condition = _condition_number(self.x_matrix)
        if math.isfinite(condition) and condition <= self.analytic_condition_threshold:
            return "analytic"
        return "latent"


    def _prepare_augmented_mu_system(self) -> None:
        n_elements = len(self.elements)
        stable_rank = self.x_rank
        n_gauge_rows = n_elements - stable_rank
        eye = torch.eye(n_elements, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)

        best_condition = float("inf")
        best_stable_indices: tuple[int, ...] | None = None
        best_gauge_indices: tuple[int, ...] | None = None
        best_augmented_matrix: torch.Tensor | None = None

        stable_index_candidates = itertools.combinations(
            range(self.x_matrix.shape[0]),
            stable_rank,
        )
        gauge_index_candidates = tuple(
            itertools.combinations(range(n_elements), n_gauge_rows)
        )
        for stable_indices in stable_index_candidates:
            stable_rows = self.x_matrix[list(stable_indices)]
            if _matrix_rank(stable_rows) != stable_rank:
                continue
            for gauge_indices in gauge_index_candidates:
                gauge_rows = eye[list(gauge_indices)]
                augmented_matrix = torch.cat([stable_rows, gauge_rows], dim=0)
                if _matrix_rank(augmented_matrix) != n_elements:
                    continue
                condition = _condition_number(augmented_matrix)
                if condition < best_condition:
                    best_condition = condition
                    best_stable_indices = tuple(stable_indices)
                    best_gauge_indices = tuple(gauge_indices)
                    best_augmented_matrix = augmented_matrix

        if (
            best_stable_indices is None
            or best_gauge_indices is None
            or best_augmented_matrix is None
        ):
            raise RuntimeError(
                "Could not construct a full-rank augmented chemical-potential "
                "system."
            )

        self.independent_stable_row_indices = best_stable_indices
        self.gauge_component_indices = best_gauge_indices
        self.augmented_x_matrix = best_augmented_matrix
        self.gauge_x_matrix = eye[list(self.gauge_component_indices)]
        self.augmented_condition = best_condition


    def _stable_gibbs_vector(self, system: ThermodynamicSystem) -> torch.Tensor:
        values = []
        for phase, composition in zip(
            self.equilibrium.phases,
            self.equilibrium.phase_compositions,
            strict=True,
        ):
            values.append(
                system.get_gibbs_energy(
                    phase,
                    composition,
                    self.equilibrium.temperature,
                ).reshape(())
            )
        return torch.stack(values)


    def solve_mu_augmented(self, system: ThermodynamicSystem) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError("Latent chemical-potential RHS parameter is missing.")
        g_vector = self._stable_gibbs_vector(system)
        known_rhs = g_vector[list(self.independent_stable_row_indices)]
        rhs = torch.cat([known_rhs, self.mu], dim=0)
        return torch.linalg.solve(self.augmented_x_matrix, rhs)


    def initial_mu_by_minimization(self,
        system: ThermodynamicSystem,
        *,
        lr: float = 2000.0,
        max_iter: int = 1000,
        cosine_decay: bool = True,
        convergence_tol: float = 50.0,
        relu_margin: float = 0.0,
        unstable_huber_beta: float | None = 1.0,
        scale_energy_by_rt: bool = True,
        console = None,
    ):
        if self.mu is None:
            return
        if self.mu.numel() == 0:
            return

        with torch.no_grad():
            g_vector = self._stable_gibbs_vector(system)
            mu_guess = torch.linalg.pinv(self.x_matrix) @ g_vector
            self.mu.copy_(self.gauge_x_matrix @ mu_guess)

        if console is not None:
            console.rule("INITIAL CHEMICAL POTENTIAL")

        model_parameters = list(system.parameters())
        previous_requires_grad = [
            parameter.requires_grad
            for parameter in model_parameters
        ]
        for parameter in model_parameters:
            parameter.requires_grad_(False)

        try:
            optimizer = torch.optim.Adam([self.mu], lr=lr)
            scheduler = None
            if cosine_decay:
                total_steps = max(1, int(max_iter))

                def lr_factor(step: int) -> float:
                    progress = min(max(step, 0), total_steps) / total_steps
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lr_factor,
                )

            converged = False
            max_delta = float("inf")
            steps = 0
            for steps in range(1, max_iter + 1):
                previous_mu = self.mu.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                loss_parts = phase_equilibrium_loss_parts(
                    self,
                    system,
                    relu_margin=relu_margin,
                    unstable_huber_beta=unstable_huber_beta,
                    scale_energy_by_rt=scale_energy_by_rt,
                )
                loss = (
                    loss_parts.stable_loss
                    + loss_parts.unstable_loss
                )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                max_delta = float((self.mu.detach() - previous_mu).abs().max().cpu())
                if max_delta <= convergence_tol:
                    converged = True
                    break

            if console is not None:
                console.print(
                    f"{self.strategy} mu: "
                    f"steps={steps}, "
                    f"converged to {convergence_tol:g} J/mol: {converged}, "
                    f"max delta={max_delta:10.2e} J/mol"
                )
                current_mu = self.solve_mu_augmented(system)
                mu_text = ", ".join(
                    f"mu_{element}={_format_scalar_for_print(current_mu[index])}"
                    for index, element in enumerate(self.elements)
                )
                console.print(f"     chemical potential: {mu_text}")
        finally:
            for parameter, requires_grad in zip(
                model_parameters,
                previous_requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)


    def current_mu_dict(
        self,
        system: ThermodynamicSystem,
    ) -> Mapping[str, torch.Tensor]:
        if self.strategy == "analytic":
            # solve mu analytically
            g_vector = self._stable_gibbs_vector(system)
            if self.x_matrix.shape[0] == self.x_matrix.shape[1]:
                mu = torch.linalg.solve(self.x_matrix, g_vector)
            else:
                mu = torch.linalg.lstsq(self.x_matrix, g_vector).solution
        else:
            # latent mu
            if self.mu is None:
                raise RuntimeError("Latent chemical potential parameter is missing.")
            mu = self.solve_mu_augmented(system)

        return {
            element: mu[index]
            for index, element in enumerate(self.elements)
        }


    def forward(self) -> torch.Tensor:
        raise RuntimeError(
            "PhaseEquilibriumOptState requires a ThermodynamicSystem. "
            "Use phase_equilibrium_loss_parts(state, system, ...) instead."
        )


def print_phi_at_equilibria(
    loss: EquilibriumLossRecord,
    *,
    console=None
) -> None:
    if console is None:
        from rich.console import Console
        console = Console()
        
    console.print(f"{loss.mu_strategy} mu: {str(loss.equilibrium)}")
    mu_text = ", ".join(
        f"mu_{element}={_format_scalar_for_print(value)}"
        for element, value in loss.mu.items()
    )
    console.print(f"     chemical potential: {mu_text}")
    for phase_id, phi in loss.phi.items():
        marker = " <- stable" if phase_id in loss.stable_phase_ids else ""
        console.print(
            f"     phi({str(phase_id):>10s}) = "
            f"{_format_scalar_for_print(phi)}{marker}"
        )


def phase_equilibrium_loss_parts(
    state: PhaseEquilibriumOptState,
    system: ThermodynamicSystem,
    *,
    relu_margin: float = 0.0,
    unstable_huber_beta: float | None = 1.0,
    scale_energy_by_rt: bool = True,
) -> EquilibriumLossRecord:
    temperature = torch.as_tensor(
        state.equilibrium.temperature,
        device=DEFAULT_DEVICE,
        dtype=DEFAULT_TYPE,
    )
    if scale_energy_by_rt:
        rt = R * temperature
        energy_scale = float(rt.detach().cpu())
    else:
        rt = 1.0
        energy_scale = 1.0
    mu_dict = state.current_mu_dict(system)

    phases_to_evaluate = tuple(system.get_competing_phases(state.elements))
    phi_by_id = {
        phase: system.get_grand_potential(
            phase,
            mu_dict,
            temperature,
        )
        for phase in phases_to_evaluate
    }

    phi_observed = [
        phi_by_id[phase]
        for phase in state.equilibrium.phases
    ]
    phi_all = list(phi_by_id.values())

    stable_total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if phi_observed:
        stable_total = (
            torch.stack(phi_observed) / rt
        ).square().sum()

    unstable_total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if phi_all:
        unstable_violation = F.relu(
            (relu_margin - torch.stack(phi_all)) / rt
        )
        if unstable_huber_beta is None:
            unstable_penalty = unstable_violation.sum()
        else:
            unstable_penalty = F.smooth_l1_loss(
                unstable_violation,
                torch.zeros_like(unstable_violation),
                beta=unstable_huber_beta / energy_scale,
                reduction="sum",
            )
        unstable_total = unstable_penalty

    return EquilibriumLossRecord(
        equilibrium=state.equilibrium,
        mu_strategy=state.strategy,
        mu=mu_dict,
        phi=phi_by_id,
        stable_phase_ids=state.stable_phase_ids,
        stable_loss=stable_total,
        unstable_loss=unstable_total,
    )
