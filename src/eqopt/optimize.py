from __future__ import annotations

from typing import Mapping, Sequence
import torch
from torch import Tensor
import torch.nn.functional as F

from .models import RedlichKisterModel, GibbsModel
from .utilities import as_float_tensor, R
from .tdb_reference import EquilibriumCompositions, PhaseCompositions


def phase_name(item: PhaseCompositions) -> str:
    return item.name


def phase_x(
    item: PhaseCompositions,
    elements: Sequence[str],
) -> list[float]:
    return [item.compositions.get(element, 0.0) for element in elements]


def solve_chemical_potential(
    phases: Mapping[str, GibbsModel],
    observed: Sequence[PhaseCompositions],
    temperature,
    *,
    ridge: float = 1.0e-10,
) -> Tensor:
    """Solve X mu = G for chemical potentials from observed phase vertices."""
    first_phase = next(iter(phases.values()))
    rows = []
    values = []
    for item in observed:
        phase = phases[phase_name(item)]
        x = as_float_tensor(
            phase_x(item, phase.elements),
            device=phase.device,
            dtype=phase.dtype,
        )
        rows.append(x / x.sum())
        values.append(phase(rows[-1], temperature).reshape(()))

    x_matrix = torch.stack(rows)
    g_vector = torch.stack(values)
    if x_matrix.shape[0] == x_matrix.shape[1]:
        return torch.linalg.solve(x_matrix, g_vector)

    eye = torch.eye(
        first_phase.n_components, device=first_phase.device, dtype=first_phase.dtype
    )
    lhs = x_matrix.T @ x_matrix + ridge * eye
    rhs = x_matrix.T @ g_vector

    return torch.linalg.solve(lhs, rhs)


def phase_equilibrium_loss_parts(
    phases: Mapping[str, GibbsModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    n_samples: int = 128,
    tau: float = 1.0,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 0.0,
) -> dict[str, Tensor]:
    """Return named loss contributions for phase equilibrium.

    Observed phases are penalized toward Phi / RT = 0.
    Unstable phases are penalized when their Phi drops below zero.
    """
    first_phase = next(iter(phases.values()))
    stable_total = torch.zeros((), device=first_phase.device, dtype=first_phase.dtype)
    unstable_total = torch.zeros((), device=first_phase.device, dtype=first_phase.dtype)
    regularization = torch.zeros((), device=first_phase.device, dtype=first_phase.dtype)

    for eq in equilibria:
        temperature = as_float_tensor(
            eq.temperature, device=first_phase.device, dtype=first_phase.dtype
        )
        rt = R * temperature

        mu = solve_chemical_potential(phases, eq.phases, temperature)
        phi_by_name = {
            name: phase.grand_potential(
                mu.flatten(),
                temperature,
                tau=tau,
                n_samples_each_side=n_samples,
            )
            for name, phase in phases.items()
        }

        observed_phases = [phase_name(item) for item in eq.phases]

        stable_losses = [
            (phi_by_name[phase_name] / rt).square() for phase_name in observed_phases
        ]
        unstable_losses = [
            F.relu(-phi / rt) for phi in phi_by_name.values()
            #if phase_name not in observed_phases
            # even though we already penalized stable phase, we add it to unstable losses
        ]

        if stable_losses:
            stable_total = stable_total + stable_weight * torch.stack(stable_losses).sum()
        if unstable_losses:
            unstable_total = unstable_total + unstable_weight * torch.stack(unstable_losses).sum()

    if regularization_weight:
        parameters = [
            parameter
            for phase in phases.values()
            for parameter in phase.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            squared_sum = sum(
                (parameter.square().sum() for parameter in parameters),
                torch.zeros((), device=first_phase.device, dtype=first_phase.dtype),
            )
            n_parameters = sum(parameter.numel() for parameter in parameters)
            regularization = regularization_weight * squared_sum / n_parameters

    normalizer = max(len(equilibria), 1)
    stable_total = stable_total / normalizer
    unstable_total = unstable_total / normalizer
    regularization = regularization / normalizer
    total = stable_total + unstable_total + regularization

    return {
        "stable": stable_total,
        "unstable": unstable_total,
        "regularization": regularization,
        "total": total,
    }


def optimize_thermodynamic_parameters(
    phases: Mapping[str, GibbsModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    steps: int = 1000,
    lr: float = 1.0e-3,
    n_samples: int = 128,
    tau: float = 1.0,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 0.0,
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
    console.print(f'regularization weights = {regularization_weight}')
    console.print(f'optimizer = {optimizer_cls}')
    
    parameters = [parameter for phase in phases.values() for parameter in phase.parameters()]
    if not parameters:
        raise ValueError("No trainable parameters found in the supplied phases.")
    
    console.rule('PHASES')
    for phase_name, model in phases.items():
        _np = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f'{phase_name:<20s} ({_np:d} parameters)')

    console.rule('EQUILIBRIA')
    for ieq, eq in enumerate(equilibria):
        console.print(f'{ieq:3d}) {eq}')
    
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
    
    console.print(f"initial loss: {history[0]:.2f}")
    console.print(f"final   loss: {history[-1]:.2f}")
    console.rule('FINISHED')
    return history


def _demo():
    torch.manual_seed(0)
    phases = {
        "alpha": RedlichKisterModel(2, polynomial_order=1, interaction_order=1, name="alpha"),
        "beta": RedlichKisterModel(2, polynomial_order=1, interaction_order=1, name="beta"),
    }
    data = [
        EquilibriumCompositions(
            temperature=1000.0,
            phases=[
                PhaseCompositions("alpha", {"C0": 0.20, "C1": 0.80}),
                PhaseCompositions("beta", {"C0": 0.75, "C1": 0.25}),
            ],
        )
    ]
    losses = optimize_thermodynamic_parameters(phases, data, steps=1000, lr=1.0)
    print(f"initial loss: {losses[0]:.6g}")
    print(f"final loss:   {losses[-1]:.6g}")
