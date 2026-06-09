import dataclasses
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
    return f"{value:10.2e}"


class PhaseEquilibriumLoss:
    def __init__(self, 
        all_phases: Sequence[PhaseEntry],
        stable_weight: float = 1.0,
        unstable_weight: float = 1.0,
        regularization_weight: float = 1e-12,
        regularize_difference: bool = False,
        *,
        n_samples: int = 64,
        tau: float = None,
        relu_margin: float = 0.0,
        unstable_huber_beta: float | None = 1.0,
        n_steps: int = 6,
        delta: float = 0.3,
        ):
        _phase_name_counts = Counter([phase.phase_name for phase in all_phases])
        for phase_name, counts in _phase_name_counts.items():
            if counts > 1:
                raise ValueError(f'phase {phase_name} is not unique, appeared {counts} times')
            
        self.all_phases = all_phases
        self.stable_weight = stable_weight
        self.unstable_weight = unstable_weight
        self.regularization_weight = regularization_weight

        if regularize_difference:
            self.parameter0 = _snapshot_trainable_parameters(all_phases)
        else:
            self.parameter0 = None

        self.n_samples_each_side = n_samples
        self.relu_margin = relu_margin
        self.unstable_huber_beta = unstable_huber_beta
        self.tau = tau
        self.n_steps = n_steps
        self.delta = delta
    

    def solve_chemical_potential(self, 
        equilibria: PhaseEquilibrium, 
        *, 
        ridge: float = 1.0e-10
    ) -> Mapping[str, torch.Tensor]:
        """Solve X mu = G for chemical potentials from observed phase vertices."""
        if len(equilibria.phases) != len(equilibria.phase_compositions):
            raise ValueError(
                "PhaseEquilibrium.phases and phase_compositions must have "
                "the same length."
            )
        rows = []
        values = []
        elements = tuple(sorted(equilibria.elements))
        for phase, composition in zip(equilibria.phases, equilibria.phase_compositions):
            x = torch.as_tensor(
                [composition.get(element, 0.0) for element in elements],
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            rows.append(x / x.sum())
            values.append(
                phase.model.gibbs_energy_per_molar_atom(
                    composition,
                    equilibria.temperature,
                ).reshape(())
            )

        x_matrix = torch.stack(rows)
        g_vector = torch.stack(values)
        if x_matrix.shape[0] == x_matrix.shape[1]:
            mu = torch.linalg.solve(x_matrix, g_vector)
        else:
            eye = torch.eye(
                len(elements),
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            lhs = x_matrix.T @ x_matrix + ridge * eye
            rhs = x_matrix.T @ g_vector
            mu = torch.linalg.solve(lhs, rhs)
        return {
            element: mu[index]
            for index, element in enumerate(elements)
        }


    def print_phi_at_equilibria(self, 
        loss_parts: Mapping[str, object], 
        *,
        console=None
    ) -> None:
        if console is None:
            from rich.console import Console
            console = Console()

        for index, entry in enumerate(loss_parts["phi_at_equilibria"]):
            eq = entry["equilibrium"]
            mu = entry["mu"]
            phi_by_id = entry["phi"]
            stable_phase_ids = entry["stable_phase_ids"]

            console.print(f"{index:3d}) {str(eq)}")
            mu_text = ", ".join(
                f"mu_{element}={_format_scalar_for_print(value)}"
                for element, value in mu.items()
            )
            console.print(f"     chemical potential: {mu_text}")
            for phase_name, phi in phi_by_id.items():
                marker = " <- stable" if phase_name in stable_phase_ids else ""
                console.print(
                    f"     phi({phase_name:>10s}) = "
                    f"{_format_scalar_for_print(phi)}{marker}"
                )


    def get_loss_parts(self, equilibria: Sequence[PhaseEquilibrium]) -> dict[str, object]:
        """calculate loss given a batch of equilibrium points"""
        stable_total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        unstable_total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        regularization = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)

        phi_at_equilibria = []
        for eq in equilibria:
            temperature = torch.as_tensor(
                eq.temperature,
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            rt = R * temperature

            mu_dict = self.solve_chemical_potential(eq)
            stable_phase_ids = {phase.phase_name for phase in eq.phases}
            other_possible_phases = _get_the_rest_of_phases(
                _get_related_phases(self.all_phases, eq.elements),
                eq.phases,
            )
            phases_to_evaluate = list(eq.phases) + list(other_possible_phases)
            phi_by_id = {
                phase.phase_name: phase.model.grand_potential_per_molar_atom(
                    mu_dict,
                    temperature,
                    tau=self.tau,
                    n_samples_each_side=self.n_samples_each_side,
                    n_steps=self.n_steps,
                    delta=self.delta,
                )
                for phase in phases_to_evaluate
            }

            phi_observed = [
                phi_by_id[phase.phase_name]
                for phase in eq.phases
            ]
            phi_all = list(phi_by_id.values())
            if phi_observed:
                stable_total = stable_total + (
                    torch.stack(phi_observed) / rt
                ).square().sum() * self.stable_weight
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
                        beta=self.unstable_huber_beta /rt,
                        reduction="sum",
                    )
                unstable_total = unstable_total + unstable_penalty * self.unstable_weight

            phi_at_equilibria.append({
                "equilibrium": eq,
                "phi": phi_by_id,
                "mu": mu_dict,
                "stable_phase_ids": stable_phase_ids,
            })
            
        if self.regularization_weight:
            regularization = self.regularization_weight * self.get_regularization_loss()

        normalizer = max(len(equilibria), 1)
        stable_total = stable_total / normalizer
        unstable_total = unstable_total / normalizer
        regularization = regularization
        total = stable_total + unstable_total + regularization

        return {
            "phi_at_equilibria": phi_at_equilibria,
            "stable": stable_total,
            "unstable": unstable_total,
            "regularization": regularization,
            "total": total,
        }
    

    def get_regularization_loss(self) -> torch.Tensor:
        total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        if self.parameter0 is None:
            """direct"""
            for phase in self.all_phases:
                for parameter in phase.model.parameters():
                    if parameter.requires_grad:
                        total = total + parameter.square().sum()
        else:
            """difference"""
            for phase in self.all_phases:
                phase_reference = self.parameter0.get(phase.phase_name)
                if phase_reference is None:
                    raise ValueError(f"Missing parameter reference for {phase.phase_name!r}.")
                for parameter_name, parameter in phase.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter_name not in phase_reference:
                        raise ValueError(
                            f"Missing parameter reference for "
                            f"{phase.phase_name}.{parameter_name}."
                        )
                    reference_parameter = phase_reference[parameter_name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    if reference_parameter.shape != parameter.shape:
                        raise ValueError(
                            f"Reference shape mismatch for "
                            f"{phase.phase_name}.{parameter_name}: expected "
                            f"{tuple(parameter.shape)}, got "
                            f"{tuple(reference_parameter.shape)}."
                        )
                    total = total + (parameter - reference_parameter).square().sum()
        return total


    def __call__(self, equilibria: Sequence[PhaseEquilibrium]) -> torch.Tensor:
        return self.get_loss_parts(equilibria)["total"]
