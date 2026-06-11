import dataclasses
import math
import torch
import torch.nn.functional as F
from collections.abc import Mapping, Set, Sequence
from collections import Counter
from ase import Atoms

from .models.models_abc import ThermodynamicModel
from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .utilities import R


@dataclasses.dataclass
class PhaseEntry:
    """we require phase_name to be unique. in order to index"""
    phase_name: str
    elements: Set[str]
    model: ThermodynamicModel
    prototype_name: str | None = None
    strukturbericht: str | None = None
    structure: Atoms | None = None


@dataclasses.dataclass
class PhaseEquilibrium:
    phases: Sequence[PhaseEntry]
    phase_compositions: Sequence[Mapping[str, float]] # ordered as phases
    temperature: float

    @property
    def elements(self) -> Set[str]:
        ele = set()
        for phase in self.phases:
            ele |= phase.elements
        return ele
    
    def __repr__(self) -> str:
        s = f'T = {self.temperature:g}, '
        parts = []
        for phase, composition in zip(self.phases, self.phase_compositions):
            sorted_ele = sorted(list(composition.keys()))
            p = f'{phase.phase_name}('
            p+= ','.join([f'x_{ele}={composition[ele]:.3f}' for ele in sorted_ele])
            p+= ')'
            parts.append(p)
        
        s += ' = '.join(parts)
        return s
    

def _get_related_phases(
    phases: Sequence[PhaseEntry], 
    given_elements: Set[str] | Sequence[str]
) -> Sequence[PhaseEntry]:
    """return the entries possible to form given elements"""
    given_elements = set(given_elements)
    return [
        phase for phase in phases if phase.elements <= given_elements
    ]


def _get_the_rest_of_phases(
    phases_all: Sequence[PhaseEntry], 
    phases_in: Sequence[PhaseEntry]
) -> Sequence[PhaseEntry]:
    """return all phases except ones in phases_in"""
    input_ids = set([p.phase_name for p in phases_in])
    return [
        phase for phase in phases_all if phase.phase_name not in input_ids
    ]


def _snapshot_trainable_parameters(phase_entries: Sequence[PhaseEntry]
) -> dict[str, dict[str, torch.Tensor]]:
    """Return detached copies of trainable parameters for displacement priors."""
    return {
        phase.phase_name: {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in phase.model.named_parameters()
            if parameter.requires_grad
        }
        for phase in phase_entries
    }


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


class SinglePhaseEquilibriumLoss(torch.nn.Module):
    def __init__(self,
        equilibrium: PhaseEquilibrium,
        all_phases: Sequence[PhaseEntry],
        *,
        n_samples: int = 64,
        tau: float | None = None,
        relu_margin: float = 0.0,
        use_tangent_huber: bool = True,
        unstable_huber_beta: float | None = 1.0,
        n_steps: int = 6,
        delta: float = 0.3,
        # mu calculation options
        mu_init_lr: float = 5000.0,
        mu_init_max_iter: int = 1000,
        mu_convergence_tol: float = 10.0,
        mu_init_cosine_decay: bool = True,
        mu_strategy: str = "auto",
        analytic_condition_threshold: float = 1.0e10,
        initialize_mu: bool = True,
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
        if use_tangent_huber and unstable_huber_beta is None:
            raise ValueError(
                "unstable_huber_beta must be set when use_tangent_huber=True."
            )

        self.equilibrium = equilibrium
        self.all_phases = tuple(all_phases)
        self.elements = tuple(sorted(equilibrium.elements))
        self.n_samples_each_side = n_samples
        self.tau = tau
        self.relu_margin = relu_margin
        self.use_tangent_huber = use_tangent_huber
        self.unstable_huber_beta = unstable_huber_beta
        self.n_steps = n_steps
        self.delta = delta
        self.analytic_condition_threshold = analytic_condition_threshold
        self.console = console
        self.stable_phase_ids = {phase.phase_name for phase in self.equilibrium.phases}
        self.phases_to_evaluate = _get_related_phases(
            self.all_phases, 
            self.equilibrium.elements
        )

        self.x_matrix = torch.stack(
            [_composition_tensor(composition, self.elements) 
            for composition in self.equilibrium.phase_compositions], dim=0
        )

        self.strategy = self._select_mu_strategy(mu_strategy)
        if self.strategy == "latent":
            self.mu = torch.nn.Parameter(
                torch.empty(
                    len(self.elements),
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

        if self.x_matrix.shape[0] != self.x_matrix.shape[1]:
            return "latent"
        condition = torch.linalg.cond(self.x_matrix.detach()).item()
        if math.isfinite(condition) and condition <= self.analytic_condition_threshold:
            return "analytic"
        return "latent"


    def _mu_dict_from_tensor(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            element: mu[index]
            for index, element in enumerate(self.elements)
        }


    def solve_mu_analytic(self, *, ridge: float=1.0e-10) -> Mapping[str, torch.Tensor]:
        values = []
        for phase, composition in zip(
            self.equilibrium.phases,
            self.equilibrium.phase_compositions,
            strict=True,
        ):
            values.append(
                phase.model.gibbs_energy_per_molar_atom(
                    composition,
                    self.equilibrium.temperature,
                ).reshape(())
            )

        g_vector = torch.stack(values)
        if self.x_matrix.shape[0] == self.x_matrix.shape[1]:
            mu = torch.linalg.solve(self.x_matrix, g_vector)
        else:
            eye = torch.eye(
                len(self.elements),
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            lhs = self.x_matrix.T @ self.x_matrix + ridge * eye
            rhs = self.x_matrix.T @ g_vector
            mu = torch.linalg.solve(lhs, rhs)
        return self._mu_dict_from_tensor(mu)


    def initial_mu_by_minimization(self,
        lr: float,
        max_iter: int,
        cosine_decay: bool,
        convergence_tol: float
    ):
        if self.mu is None:
            return

        with torch.no_grad():
            values = [
                phase.model.gibbs_energy_per_molar_atom(
                    composition,
                    self.equilibrium.temperature,
                ).reshape(())
                for phase, composition in zip(
                    self.equilibrium.phases,
                    self.equilibrium.phase_compositions,
                    strict=True,
                )
            ]
            self.mu.copy_(torch.stack(values).mean())

        if self.console is not None:
            self.console.rule("INITIAL CHEMICAL POTENTIAL")

        model_parameters = [
            parameter
            for phase in self.all_phases
            for parameter in phase.model.parameters()
        ]
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
                    loss_parts["tangent"]
                    + loss_parts["stable"]
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
                mu_text = ", ".join(
                    f"mu_{element}={_format_scalar_for_print(self.mu[index])}"
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
        return self._mu_dict_from_tensor(self.mu)


    def get_loss_parts(self) -> dict[str, object]:
        temperature = torch.as_tensor(
            self.equilibrium.temperature,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        rt = R * temperature
        mu_dict = self._current_mu_dict()
        mu = torch.stack([mu_dict[element] for element in self.elements])

        tangent_total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        if self.strategy == 'latent':
            """this part will be zero if use analytic loss"""
            for phase, composition, x in zip(
                self.equilibrium.phases,
                self.equilibrium.phase_compositions,
                self.x_matrix,
                strict=True,
            ):
                gibbs = phase.model.gibbs_energy_per_molar_atom(
                    composition,
                    self.equilibrium.temperature,
                ).reshape(())
                tangent_residual = (gibbs - x @ mu) / rt
                if not self.use_tangent_huber:
                    tangent_penalty = tangent_residual.square()
                else:
                    tangent_penalty = F.smooth_l1_loss(
                        tangent_residual,
                        torch.zeros_like(tangent_residual),
                        beta=self.unstable_huber_beta / rt,
                        reduction="sum",
                    )
                tangent_total = tangent_total + tangent_penalty

        phi_by_id = {
            phase.phase_name: phase.model.grand_potential_per_molar_atom(
                mu_dict,
                temperature,
                tau=self.tau,
                n_samples_each_side=self.n_samples_each_side,
                n_steps=self.n_steps,
                delta=self.delta,
            )
            for phase in self.phases_to_evaluate
        }

        phi_observed = [
            phi_by_id[phase.phase_name]
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
                    beta=self.unstable_huber_beta / rt,
                    reduction="sum",
                )
            unstable_total = unstable_penalty

        return {
            "equilibrium": self.equilibrium,
            "mu_strategy": self.strategy,
            "mu": mu_dict,
            "phi": phi_by_id,
            "stable_phase_ids": self.stable_phase_ids,
            "tangent": tangent_total,
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
        for phase_name, phi in loss_parts["phi"].items():
            marker = " <- stable" if phase_name in loss_parts["stable_phase_ids"] else ""
            console.print(
                f"     phi({phase_name:>10s}) = "
                f"{_format_scalar_for_print(phi)}{marker}"
            )


    def forward(self) -> torch.Tensor:
        """this is defined but should not be used since there is no weights"""
        loss_dict = self.get_loss_parts()
        return loss_dict["tangent"] + loss_dict['stable'] + loss_dict['unstable']


