from __future__ import annotations

from typing import Mapping, Sequence
import torch
from torch import Tensor
import torch.nn.functional as F

from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .models import ThermodynamicModel
from .utilities import R
from .tdb_reference import EquilibriumCompositions, PhaseCompositions


def phase_name(item: PhaseCompositions) -> str:
    return item.name


def phase_x(
    item: PhaseCompositions,
    elements: Sequence[str],
) -> list[float]:
    return [item.compositions.get(element, 0.0) for element in elements]


def solve_chemical_potential(
    phases: Mapping[str, ThermodynamicModel],
    observed: Sequence[PhaseCompositions],
    temperature,
    *,
    ridge: float = 1.0e-10,
) -> Tensor:
    """Solve X mu = G for chemical potentials from observed phase vertices."""
    first_phase = next(iter(phases.values()))
    elements = first_phase.elements
    rows = []
    values = []
    for item in observed:
        phase = phases[phase_name(item)]
        x = torch.as_tensor(
            phase_x(item, elements),
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        rows.append(x / x.sum())
        values.append(
            phase.gibbs_energy_per_molar_atom(
                item.compositions,
                temperature,
            ).reshape(())
        )

    x_matrix = torch.stack(rows)
    g_vector = torch.stack(values)
    if x_matrix.shape[0] == x_matrix.shape[1]:
        return torch.linalg.solve(x_matrix, g_vector)

    eye = torch.eye(
        len(elements),
        device=DEFAULT_DEVICE,
        dtype=DEFAULT_TYPE,
    )
    lhs = x_matrix.T @ x_matrix + ridge * eye
    rhs = x_matrix.T @ g_vector

    return torch.linalg.solve(lhs, rhs)


ParameterReference = Mapping[str, Mapping[str, Tensor]]


def snapshot_trainable_parameters(
    phases: Mapping[str, ThermodynamicModel],
) -> dict[str, dict[str, Tensor]]:
    """Return detached copies of trainable parameters for displacement priors."""
    return {
        phase_name: {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in phase.named_parameters()
            if parameter.requires_grad
        }
        for phase_name, phase in phases.items()
    }


def parameter_change_l2_sum(
    phases: Mapping[str, ThermodynamicModel],
    reference: ParameterReference,
) -> Tensor:
    """Return sum_i |w_i - w_i0|^2 over all trainable phase parameters."""
    total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    for phase_name, phase in phases.items():
        if phase_name not in reference:
            raise ValueError(f"Missing parameter reference for phase {phase_name!r}.")
        phase_reference = reference[phase_name]
        for parameter_name, parameter in phase.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter_name not in phase_reference:
                raise ValueError(
                    f"Missing parameter reference for {phase_name}.{parameter_name}."
                )
            reference_parameter = phase_reference[parameter_name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if reference_parameter.shape != parameter.shape:
                raise ValueError(
                    f"Reference shape mismatch for {phase_name}.{parameter_name}: "
                    f"expected {tuple(parameter.shape)}, got "
                    f"{tuple(reference_parameter.shape)}."
                )
            total = total + (parameter - reference_parameter).square().sum()
    return total


def phase_equilibrium_loss_parts(
    phases: Mapping[str, ThermodynamicModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    n_samples: int = 128,
    tau: float = 1.0,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 0.0,
    parameter_reference: ParameterReference | None = None,
    relu_margin: float = 1.0,
) -> dict[str, Tensor]:
    """Return named loss contributions for phase equilibrium.

    Observed phases are penalized toward Phi / RT = 0.
    Unstable phases are penalized when their Phi drops below zero.
    """
    first_phase = next(iter(phases.values()))
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

        mu = solve_chemical_potential(phases, eq.phases, temperature)
        mu_dict = {
            element: mu[index]
            for index, element in enumerate(first_phase.elements)
        }
        phi_by_name = {
            name: phase.grand_potential_per_molar_atom(
                mu_dict,
                temperature,
                tau=tau,
                n_samples_each_side=n_samples,
            )
            for name, phase in phases.items()
        }
        phi_at_equilibria.append({
            "equilibrium": eq, "phi": phi_by_name, 'mu': mu
        })

        observed_phases = [phase_name(item) for item in eq.phases]

        stable_losses = [
            (phi_by_name[phase_name] / rt).square() for phase_name in observed_phases
        ]
        unstable_losses = [
            F.relu((relu_margin-phi) / rt) for phi in phi_by_name.values()
            #if phase_name not in observed_phases
            # even though we already penalized stable phase, we add it to unstable losses
        ]

        if stable_losses:
            stable_total = stable_total + stable_weight * torch.stack(stable_losses).sum()
        if unstable_losses:
            unstable_total = unstable_total + unstable_weight * torch.stack(unstable_losses).sum()

    if regularization_weight:
        if parameter_reference is None:
            parameter_reference = snapshot_trainable_parameters(phases)
        regularization = (
            regularization_weight
            * parameter_change_l2_sum(phases, parameter_reference)
        )

    normalizer = max(len(equilibria), 1)
    stable_total = stable_total / normalizer
    unstable_total = unstable_total / normalizer
    regularization = regularization / normalizer
    total = stable_total + unstable_total + regularization

    return {
        "phi_at_equilibria": phi_at_equilibria,
        "stable": stable_total,
        "unstable": unstable_total,
        "regularization": regularization,
        "total": total,
    }


def _format_scalar_for_print(value: Tensor | float) -> str:
    if isinstance(value, Tensor):
        value = float(value.detach().cpu().reshape(()))
    return f"{value:10.2e}"


def print_phi_at_equilibria(
    loss_parts: Mapping[str, object],
    phases: Mapping[str, ThermodynamicModel],
    *,
    console=None,
) -> None:
    """Print chemical potentials and phase grand potentials for each equilibrium."""
    if console is None:
        from rich.console import Console
        console = Console()

    first_phase = next(iter(phases.values()))
    elements = first_phase.elements

    for index, entry in enumerate(loss_parts["phi_at_equilibria"]):
        eq = entry["equilibrium"]
        mu = entry["mu"].detach().cpu().reshape(-1)
        phi_by_name = entry["phi"]

        console.print(f"{index:3d}) {str(eq)}")

        if len(elements) == mu.numel():
            mu_text = ", ".join(
                f"mu_{element}={float(value):.3e}"
                for element, value in zip(elements, mu)
            )
        else:
            mu_text = ", ".join(
                f"mu[{i}]={float(value):.3e}" for i, value in enumerate(mu)
            )
        console.print(f"     chemical potential: {mu_text}")
        observed_phases = [phase_name(item) for item in eq.phases]
        for _phase_name, phi in phi_by_name.items():
            if _phase_name in observed_phases:
                console.print(f"     phi({_phase_name:>10s}) = {_format_scalar_for_print(phi)} <- stable")
            else:
                console.print(f"     phi({_phase_name:>10s}) = {_format_scalar_for_print(phi)}")


def optimize_thermodynamic_parameters(
    phases: Mapping[str, ThermodynamicModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    steps: int = 1000,
    lr: float = 1.0e-3,
    n_samples: int = 128,
    tau: float = 0.1,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = None,
    optimizer_cls=torch.optim.Adam,
    print_every: int = 20,
    loss_threshold: float | None = None,
) -> list[float]:
    """Optimize model parameters against observed phase compositions.

    Parameters
    ----------
    steps: int
        maximal number of optimization steps
    lr: float
        learning rate
    n_samples: int
        sample density of degrees of freedome
    tau: float
        softmin parameter, the smaller tau, the more accurate min
    stable_weight: float
        weights for observed phases
    unstable_weight: float
        weights for unstable phases (ReLU loss)
    regularization_weight: float
        weights for regularization loss
    """
    from rich.console import Console
    console = Console()
    console.rule('PARAMETERS')
    console.print(f'steps = {steps}')
    if loss_threshold is not None:
        console.print(f'loss threshold = {loss_threshold}')
    console.print(f'lr = {lr}')
    console.print(f'sampling density = {n_samples}')
    console.print(f'tau (softmin) = {tau}')
    console.print(f'stable weights = {stable_weight}')
    console.print(f'unstable weights = {unstable_weight}')
    average_T = sum([eq.temperature for eq in equilibria])/len(equilibria)
    if regularization_weight is None:
        _sigma = 50
        _s = 3000
        regularization_weight = (_sigma/_s/R/average_T)**2 * stable_weight
        console.print(f'regularization weights = {regularization_weight} (automatically set)')
    else:
        console.print(f'regularization weights = {regularization_weight}')
    console.print(f'optimizer = {optimizer_cls}')
    console.print(f'average temp of equilibria = {average_T}')
    
    parameters = [parameter for phase in phases.values() for parameter in phase.parameters()]
    if not parameters:
        raise ValueError("No trainable parameters found in the supplied phases.")
    parameter_reference = snapshot_trainable_parameters(phases)
    
    console.rule('PHASES')
    for phase_name, model in phases.items():
        _np = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f'{phase_name:<20s} ({_np:d} parameters)')

    console.rule('EQUILIBRIA')
    for ieq, eq in enumerate(equilibria):
        console.print(f'{ieq:3d}) {eq}')
    with torch.no_grad():
        initial_loss_parts = phase_equilibrium_loss_parts(
            phases,
            equilibria,
            n_samples=n_samples,
            tau=tau,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
            parameter_reference=parameter_reference,
        )
    console.rule('PHI AT EQUILIBRIA (INITIAL)')
    print_phi_at_equilibria(initial_loss_parts, phases, console=console)

    optimizer = optimizer_cls(parameters, lr=lr)

    history: list[float] = []
    console.rule('OPTIMIZE')
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss_parts = phase_equilibrium_loss_parts(
            phases,
            equilibria,
            n_samples=n_samples,
            tau=tau,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
            parameter_reference=parameter_reference,
        )
        loss = loss_parts["total"]
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        
        if print_every and (len(history) == 1 or len(history) % print_every == 0):
            stable_loss = float(loss_parts["stable"].detach().cpu())
            unstable_loss = float(loss_parts["unstable"].detach().cpu())
            regularization_loss = float(loss_parts["regularization"].detach().cpu())
            console.print(
                f"step {len(history):>6d}/{steps}: "
                f"loss={history[-1]:10.2e}, "
                f"stable={stable_loss:10.2e}, "
                f"unstable={unstable_loss:10.2e}, "
                f"regularization={regularization_loss:10.2e}"
            )
        if loss_threshold is not None and history[-1] <= loss_threshold:
            if print_every:
                console.print(
                    "\n"
                    f"stopping early at step {len(history)}/{steps}: "
                    f"loss={history[-1]:.2e} <= threshold={loss_threshold:.2e}"
                )
            break

    with torch.no_grad():
        final_loss_parts = phase_equilibrium_loss_parts(
            phases,
            equilibria,
            n_samples=n_samples,
            tau=tau,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
            parameter_reference=parameter_reference,
        )

    final_loss = float(final_loss_parts["total"].detach().cpu())
    console.print(f"initial loss: {history[0]:.2e}")
    console.print(f"final   loss: {final_loss:.2e}")
    console.rule('PHI AT EQUILIBRIA (FINAL)')
    print_phi_at_equilibria(final_loss_parts, phases, console=console)
    console.rule('FINISHED')
    return history
