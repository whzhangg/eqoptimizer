from typing import Sequence
import math
import torch
from .loss_function import PhaseEquilibrium, PhaseEquilibriumLoss


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Disable optimization of all parameters in a torch model."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def optimize_thermodynamic_parameters(
    loss: PhaseEquilibriumLoss,
    all_equilibria: Sequence[PhaseEquilibrium],
    *,
    batch_size: int | None = None,
    epochs: int = 100,
    lr: float = 1.0,
    optimizer_cls=torch.optim.Adam,
    print_every: int = 20,
    loss_threshold: float | None = None,
    cosine_decay: bool = False,
    min_lr_factor: float = 0.0,
):
    """Optimize thermodynamic models using PhaseEquilibriumLoss."""
    from rich.console import Console
    console = Console()

    if not all_equilibria:
        raise ValueError("No equilibria supplied for optimization.")

    parameters = [
        parameter
        for phase in loss.all_phases
        for parameter in phase.model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("No trainable parameters found in the supplied phases.")

    if batch_size is None:
        batch_size = len(all_equilibria)
    batch_size = max(1, min(int(batch_size), len(all_equilibria)))
    batches_per_epoch = math.ceil(len(all_equilibria) / batch_size)
    total_steps = max(1, int(epochs) * batches_per_epoch)

    console.rule('PARAMETERS')
    console.print(f'epochs = {epochs}')
    if loss_threshold is not None:
        console.print(f'loss threshold = {loss_threshold}')
    console.print(f'lr = {lr}')
    if cosine_decay:
        console.print(f'lr schedule = cosine decay to {min_lr_factor:g} * lr')
    console.print(f'batch size = {batch_size}')
    console.print(f'sampling density = {loss.n_samples_each_side}')
    console.print(f'tau = {loss.tau}')
    console.print(f'exp-gradient steps = {loss.n_steps}')
    console.print(f'exp-gradient delta = {loss.delta}')
    console.print(f'stable weights = {loss.stable_weight}')
    console.print(f'unstable weights = {loss.unstable_weight}')
    console.print(f'regularization weights = {loss.regularization_weight}')
    console.print(f'optimizer = {optimizer_cls}')
    average_T = sum(eq.temperature for eq in all_equilibria) / len(all_equilibria)
    console.print(f'average temp of equilibria = {average_T}')

    console.rule('PHASES')
    for phase in loss.all_phases:
        n_parameters = sum(
            parameter.numel()
            for parameter in phase.model.parameters()
            if parameter.requires_grad
        )
        console.print(f'{phase.phase_name:<20s} ({n_parameters:d} parameters)')

    console.rule('EQUILIBRIA')
    for index, eq in enumerate(all_equilibria):
        console.print(f'{index:3d}) {eq}')

    with torch.no_grad():
        initial_loss_parts = loss.get_loss_parts(all_equilibria)
    console.rule('PHI AT EQUILIBRIA (INITIAL)')
    loss.print_phi_at_equilibria(initial_loss_parts, console=console)

    optimizer = optimizer_cls(parameters, lr=lr)
    scheduler = None
    if cosine_decay:
        min_lr_factor = float(min_lr_factor)
        if min_lr_factor < 0.0 or min_lr_factor > 1.0:
            raise ValueError("min_lr_factor must be between 0 and 1.")

        def cosine_lr_factor(step: int) -> float:
            progress = min(max(step, 0), total_steps) / total_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cosine_lr_factor,
        )

    history: list[float] = []
    console.rule('OPTIMIZE')
    global_step = 0

    for epoch in range(1, epochs + 1):
        if batch_size == len(all_equilibria):
            batches = [all_equilibria]
        else:
            shuffled_indices = torch.randperm(len(all_equilibria)).tolist()
            batches = [
                [all_equilibria[index] for index in shuffled_indices[start:start + batch_size]]
                for start in range(0, len(shuffled_indices), batch_size)
            ]

        for batch in batches:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            loss_parts = loss.get_loss_parts(batch)
            total_loss = loss_parts["total"]
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            history.append(float(total_loss.detach().cpu()))

            if print_every and (global_step == 1 or global_step % print_every == 0):
                stable_loss = float(loss_parts["stable"].detach().cpu())
                unstable_loss = float(loss_parts["unstable"].detach().cpu())
                regularization_loss = float(loss_parts["regularization"].detach().cpu())
                current_lr = optimizer.param_groups[0]["lr"]
                console.print(
                    f"epoch {epoch:>4d}/{epochs}, "
                    f"step {global_step:>6d}: "
                    f"lr={current_lr:10.2e}, "
                    f"loss={history[-1]:10.2e}, "
                    f"stable={stable_loss:10.2e}, "
                    f"unstable={unstable_loss:10.2e}, "
                    f"regularization={regularization_loss:10.2e}"
                )

            if loss_threshold is not None and history[-1] <= loss_threshold:
                if print_every:
                    console.print(
                        "\n"
                        f"stopping early at epoch {epoch}/{epochs}, "
                        f"step {global_step}: "
                        f"loss={history[-1]:.2e} <= threshold={loss_threshold:.2e}"
                    )
                break
        if loss_threshold is not None and history and history[-1] <= loss_threshold:
            break

    with torch.no_grad():
        final_loss_parts = loss.get_loss_parts(all_equilibria)
    final_loss = float(final_loss_parts["total"].detach().cpu())

    if history:
        console.print(f"initial loss: {float(initial_loss_parts['total'].detach().cpu()):.2e}")
        console.print(f"final   loss: {final_loss:.2e}")
    console.rule('PHI AT EQUILIBRIA (FINAL)')
    loss.print_phi_at_equilibria(final_loss_parts, console=console)
    console.rule('FINISHED')
    return history
