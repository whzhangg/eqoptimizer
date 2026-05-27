from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .models import SolidSolutionModel, GibbsModel
from .utilities import as_float_tensor, R, simplex_samples_dirichlet
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


def phase_equilibrium_loss(
    phases: Mapping[str, GibbsModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    samples: Tensor | None = None,
    n_samples: int = 256,
    tau: float = 1.0,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 0.0,
) -> Tensor:
    """Driving-force loss from observed phase compositions.

    Observed phases are penalized toward Phi / RT = 0. All supplied phases are
    also penalized when their soft-min grand potential drops below zero.
    """
    first_phase = next(iter(phases.values()))
    if samples is None:
        samples = simplex_samples_dirichlet(
            first_phase.n_components,
            n_samples_each_side=n_samples,
            device=first_phase.device,
            dtype=first_phase.dtype,
        )

    total = torch.zeros((), device=first_phase.device, dtype=first_phase.dtype)
    for eq in equilibria:
        temperature = eq.temperature
        observed = eq.phases
        temperature = as_float_tensor(
            temperature, device=first_phase.device, dtype=first_phase.dtype
        )
        rt = R * temperature
        mu = solve_chemical_potential(phases, observed, temperature)
        phi_by_name = {
            name: phase.grand_potential(mu.flatten(), temperature, samples, tau=tau)
            for name, phase in phases.items()
        }

        stable_terms = []
        for item in observed:
            stable_terms.append((phi_by_name[phase_name(item)] / rt).square())
        if stable_terms:
            total = total + stable_weight * torch.stack(stable_terms).sum()

        unstable_terms = [
            F.relu(-phi / rt) for phi in phi_by_name.values()
        ]
        if unstable_terms:
            total = total + unstable_weight * torch.stack(unstable_terms).sum()
        
    if regularization_weight:
        reg = sum(
            (parameter.square().mean() for phase in phases.values() for parameter in phase.parameters()),
            torch.zeros((), device=first_phase.device, dtype=first_phase.dtype),
        )
        total = total + regularization_weight * reg
    
    return total / max(len(equilibria), 1)


def optimize_thermodynamic_parameters(
    phases: Mapping[str, GibbsModel],
    equilibria: Sequence[EquilibriumCompositions],
    *,
    steps: int = 1000,
    lr: float = 1.0e-3,
    n_samples: int = 256,
    tau: float = 1.0,
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 0.0,
    optimizer_cls=torch.optim.Adam,
    print_every: int = 20,
    loss_threshold: float | None = None,
) -> list[float]:
    """Optimize model parameters against observed phase compositions."""
    parameters = [parameter for phase in phases.values() for parameter in phase.parameters()]
    if not parameters:
        raise ValueError("No trainable parameters found in the supplied phases.")
    optimizer = optimizer_cls(parameters, lr=lr)
    first_phase = next(iter(phases.values()))
    samples = simplex_samples_dirichlet(
        first_phase.n_components,
        n_samples_each_side=n_samples,
        device=first_phase.device,
        dtype=first_phase.dtype,
    )

    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = phase_equilibrium_loss(
            phases,
            equilibria,
            samples=samples,
            tau=tau,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if print_every and (len(history) == 1 or len(history) % print_every == 0):
            print(f"step {len(history):>6d}/{steps}: loss={history[-1]:.6g}")
        if loss_threshold is not None and history[-1] <= loss_threshold:
            if print_every:
                print(
                    f"stopping early at step {len(history)}/{steps}: "
                    f"loss={history[-1]:.6g} <= threshold={loss_threshold:.6g}"
                )
            break

    return history


if __name__ == "__main__":
    torch.manual_seed(0)
    phases = {
        "alpha": SolidSolutionModel(2, polynomial_order=1, interaction_order=1, name="alpha"),
        "beta": SolidSolutionModel(2, polynomial_order=1, interaction_order=1, name="beta"),
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
