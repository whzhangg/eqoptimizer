import itertools
import math
import torch
import torch.nn.functional as F
from collections.abc import Mapping, Sequence

from .models.system_abc import ThermodynamicSystem
from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .utilities import R
from .phase import PhaseEquilibrium


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


class SinglePhaseEquilibriumLoss(torch.nn.Module):
    analytic_condition_threshold: float = 1.0e10
    def __init__(self,
        equilibrium: PhaseEquilibrium,
        system: ThermodynamicSystem,
        *,
        n_samples: int = 64,
        tau: float | None = None,
        use_softmin: bool = True,
        n_steps: int = 6,
        delta: float = 0.3,
        # loss calculation
        relu_margin: float = 0.0,
        unstable_huber_beta: float | None = 1.0,
        scale_energy_by_rt: bool = True,
        # mu calculation options
        mu_init_lr: float = 5000.0,
        mu_init_max_iter: int = 1000,
        mu_convergence_tol: float = 10.0,
        mu_init_cosine_decay: bool = True,
        mu_strategy: str = "auto",
        initialize_mu: bool = True,
        # IO
        console = None
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
        object.__setattr__(self, "system", system)
        self.elements = tuple(sorted(equilibrium.chemical_system))
        self.n_samples_each_side = n_samples
        self.tau = tau
        self.use_softmin = use_softmin
        self.relu_margin = relu_margin
        self.unstable_huber_beta = unstable_huber_beta
        self.scale_energy_by_rt = scale_energy_by_rt
        self.n_steps = n_steps
        self.delta = delta
        self.console = console
        self.stable_phase_ids = set(self.equilibrium.phases)
        self.phases_to_evaluate = tuple(
            self.system.get_competing_phases(self.elements)
        )

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
            if initialize_mu:
                self.initial_mu_by_minimization(
                    lr=mu_init_lr,
                    max_iter=mu_init_max_iter,
                    cosine_decay=mu_init_cosine_decay,
                    convergence_tol=mu_convergence_tol,
                )
            else:
                torch.nn.init.zeros_(self.mu)
        else:
            self.register_parameter("mu", None)

        if self.console is not None:
            self.console.print(str(self.equilibrium))
        

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


    def _mu_dict_from_tensor(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            element: mu[index]
            for index, element in enumerate(self.elements)
        }


    def _stable_gibbs_vector(self) -> torch.Tensor:
        values = []
        for phase, composition in zip(
            self.equilibrium.phases,
            self.equilibrium.phase_compositions,
            strict=True,
        ):
            values.append(
                self.system.get_gibbs_energy(
                    phase,
                    composition,
                    self.equilibrium.temperature,
                ).reshape(())
            )
        return torch.stack(values)


    def solve_mu_analytic(self) -> Mapping[str, torch.Tensor]:
        g_vector = self._stable_gibbs_vector()
        if self.x_matrix.shape[0] == self.x_matrix.shape[1]:
            mu = torch.linalg.solve(self.x_matrix, g_vector)
        else:
            mu = torch.linalg.lstsq(self.x_matrix, g_vector).solution
        return self._mu_dict_from_tensor(mu)


    def solve_mu_augmented(self) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError("Latent chemical-potential RHS parameter is missing.")
        g_vector = self._stable_gibbs_vector()
        known_rhs = g_vector[list(self.independent_stable_row_indices)]
        rhs = torch.cat([known_rhs, self.mu], dim=0)
        return torch.linalg.solve(self.augmented_x_matrix, rhs)


    def initial_mu_by_minimization(self,
        lr: float,
        max_iter: int,
        cosine_decay: bool,
        convergence_tol: float
    ):
        if self.mu is None:
            return
        if self.mu.numel() == 0:
            return

        with torch.no_grad():
            g_vector = self._stable_gibbs_vector()
            mu_guess = torch.linalg.pinv(self.x_matrix) @ g_vector
            self.mu.copy_(self.gauge_x_matrix @ mu_guess)

        if self.console is not None:
            self.console.rule("INITIAL CHEMICAL POTENTIAL")

        model_parameters = list(self.system.parameters())
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
                loss_parts = self.get_loss_parts()
                loss = (
                    loss_parts["stable"]
                    + loss_parts["unstable"]
                )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                max_delta = float((self.mu.detach() - previous_mu).abs().max().cpu())
                if max_delta <= convergence_tol:
                    converged = True
                    break

            if self.console is not None:
                self.console.print(
                    f"{self.strategy} mu: "
                    f"steps={steps}, "
                    f"converged to {convergence_tol:g} J/mol: {converged}, "
                    f"max delta={max_delta:10.2e} J/mol"
                )
                current_mu = self.solve_mu_augmented()
                mu_text = ", ".join(
                    f"mu_{element}={_format_scalar_for_print(current_mu[index])}"
                    for index, element in enumerate(self.elements)
                )
                self.console.print(f"     chemical potential: {mu_text}")
        finally:
            for parameter, requires_grad in zip(
                model_parameters,
                previous_requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)

    
    def _current_mu_dict(self) -> Mapping[str, torch.Tensor]:
        if self.strategy == "analytic":
            return self.solve_mu_analytic()
        if self.mu is None:
            raise RuntimeError("Latent chemical potential parameter is missing.")
        return self._mu_dict_from_tensor(self.solve_mu_augmented())


    def get_loss_parts(self) -> dict[str, object]:
        temperature = torch.as_tensor(
            self.equilibrium.temperature,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        if self.scale_energy_by_rt:
            rt = R * temperature
            energy_scale = float(rt.detach().cpu())
        else:
            rt = 1.0
            energy_scale = 1.0
        mu_dict = self._current_mu_dict()

        phi_by_id = {
            phase: self.system.get_grand_potential(
                phase,
                mu_dict,
                temperature,
                tau=self.tau,
                use_softmin=self.use_softmin,
                n_samples_each_side=self.n_samples_each_side,
                n_steps=self.n_steps,
                delta=self.delta,
            )
            for phase in self.phases_to_evaluate
        }

        phi_observed = [
            phi_by_id[phase]
            for phase in self.equilibrium.phases
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
                (self.relu_margin - torch.stack(phi_all)) / rt
            )
            if self.unstable_huber_beta is None:
                unstable_penalty = unstable_violation.sum()
            else:
                unstable_penalty = F.smooth_l1_loss(
                    unstable_violation,
                    torch.zeros_like(unstable_violation),
                    beta=self.unstable_huber_beta / energy_scale,
                    reduction="sum",
                )
            unstable_total = unstable_penalty

        return {
            "equilibrium": self.equilibrium,
            "mu_strategy": self.strategy,
            "mu": mu_dict,
            "phi": phi_by_id,
            "stable_phase_ids": self.stable_phase_ids,
            "stable": stable_total,
            "unstable": unstable_total,
        }


    def print_phi_at_equilibria(self, loss_parts: Mapping[str, object], *, console=None):
        if console is None:
            console = self.console
        if console is None:
            from rich.console import Console
            console = Console()
        
        console.print(f"{loss_parts['mu_strategy']} mu: {str(loss_parts['equilibrium'])}")
        mu_text = ", ".join(
            f"mu_{element}={_format_scalar_for_print(value)}"
            for element, value in loss_parts["mu"].items()
        )
        console.print(f"     chemical potential: {mu_text}")
        for phase_id, phi in loss_parts["phi"].items():
            marker = " <- stable" if phase_id in loss_parts["stable_phase_ids"] else ""
            console.print(
                f"     phi({str(phase_id):>10s}) = "
                f"{_format_scalar_for_print(phi)}{marker}"
            )


    def forward(self) -> torch.Tensor:
        """this is defined but should not be used since there is no weights"""
        loss_dict = self.get_loss_parts()
        return loss_dict['stable'] + loss_dict['unstable']
