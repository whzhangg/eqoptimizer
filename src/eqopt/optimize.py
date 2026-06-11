import dataclasses
from typing import Sequence
import math
from pathlib import Path
import time

import torch
from rich.console import Console

from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .loss_function import PhaseEntry, PhaseEquilibrium, SinglePhaseEquilibriumLoss


def get_console():
    return Console()


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Disable optimization of all parameters in a torch model."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _snapshot_trainable_model_parameters(
    all_phases: Sequence[PhaseEntry],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        phase.phase_name: {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in phase.model.named_parameters()
            if parameter.requires_grad
        }
        for phase in all_phases
    }


def _regularization_loss(
    all_phases: Sequence[PhaseEntry],
    parameter0: dict[str, dict[str, torch.Tensor]] | None,
) -> torch.Tensor:
    total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if parameter0 is None:
        for phase in all_phases:
            for parameter in phase.model.parameters():
                if parameter.requires_grad:
                    total = total + parameter.square().sum()
        return total

    for phase in all_phases:
        phase_reference = parameter0.get(phase.phase_name)
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


def _collect_trainable_parameters(
    all_phases: Sequence[PhaseEntry],
    equilibrium_losses: torch.nn.ModuleList,
) -> list[torch.nn.Parameter]:
    parameters = [
        parameter
        for phase in all_phases
        for parameter in phase.model.parameters()
        if parameter.requires_grad
    ]
    parameters.extend(
        parameter
        for parameter in equilibrium_losses.parameters()
        if parameter.requires_grad
    )
    return parameters


def _aggregate_loss_parts(
    equilibrium_losses: Sequence[SinglePhaseEquilibriumLoss],
    batch_indices: Sequence[int],
    all_phases: Sequence[PhaseEntry],
    *,
    stable_weight: float,
    unstable_weight: float,
    regularization_weight: float,
    parameter0: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, object]:
    stable = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    unstable = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    phi_at_equilibria = []

    for equilibrium_index in batch_indices:
        parts = equilibrium_losses[equilibrium_index].get_loss_parts()
        stable = stable + parts["stable"]
        unstable = unstable + parts["unstable"]
        phi_at_equilibria.append({
            "equilibrium_index": equilibrium_index,
            **parts,
        })

    normalizer = max(len(batch_indices), 1)
    stable = stable / normalizer
    unstable = unstable / normalizer
    regularization = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if regularization_weight:
        regularization = (
            regularization_weight * _regularization_loss(all_phases, parameter0)
        )

    total = (
        stable_weight * stable
        + unstable_weight * unstable
        + regularization
    )
    return {
        "phi_at_equilibria": phi_at_equilibria,
        "stable": stable,
        "unstable": unstable,
        "regularization": regularization,
        "total": total,
    }


def _print_phi_at_equilibria(
    equilibrium_losses: Sequence[SinglePhaseEquilibriumLoss],
    loss_parts: dict[str, object],
    *,
    console,
) -> None:
    for entry in loss_parts["phi_at_equilibria"]:
        equilibrium_index = entry["equilibrium_index"]
        console.print(f"{equilibrium_index:3d}) ", end="")
        equilibrium_losses[equilibrium_index].print_phi_at_equilibria(
            entry,
            console=console,
        )


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int | None = None
    epochs: int = 100
    lr: float = 1.0
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam
    print_every: int = 20
    loss_threshold: float | None = None
    cosine_decay: bool = False
    min_lr_factor: float = 0.0
    stable_weight: float = 1.0
    unstable_weight: float = 1.0
    regularization_weight: float = 1.0e-12
    regularize_difference: bool = False
    n_samples: int = 64
    tau: float | None = None
    relu_margin: float = 0.0
    unstable_huber_beta: float | None = 1.0
    n_steps: int = 6
    delta: float = 0.3
    mu_init_lr: float = 5000.0
    mu_init_max_iter: int = 1000
    mu_convergence_tol: float = 10.0
    mu_init_cosine_decay: bool = True
    mu_strategy: str = "auto"
    analytic_condition_threshold: float = 1.0e10
    print_final_results: bool = True
    mu_checkpoint_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    restart_from: str | Path | None = None
    save_every: int | None = None
    best_filename: str = "best.pt"
    last_filename: str = "last.pt"

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, object]) -> "OptimizationConfig":
        config_data = dict(state_dict.get("config", {}))
        optimizer_name = config_data.pop("optimizer_cls", "Adam")
        optimizer_cls = getattr(torch.optim, optimizer_name, torch.optim.Adam)
        field_names = {field.name for field in dataclasses.fields(cls)}
        filtered = {
            key: value
            for key, value in config_data.items()
            if key in field_names
        }
        return cls(optimizer_cls=optimizer_cls, **filtered)


@dataclasses.dataclass
class OptimizationState:
    all_phases: tuple[PhaseEntry, ...]
    all_equilibria: tuple[PhaseEquilibrium, ...]
    config: OptimizationConfig
    equilibrium_losses: torch.nn.ModuleList
    parameter0: dict[str, dict[str, torch.Tensor]] | None = None
    torch_optimizer: torch.optim.Optimizer | None = None
    scheduler: object | None = None
    history: list[float] = dataclasses.field(default_factory=list)
    epoch: int = 0
    global_step: int = 0
    initial_loss_parts: dict[str, object] | None = None
    final_loss_parts: dict[str, object] | None = None

    @classmethod
    def initialize(
        cls,
        all_phases: Sequence[PhaseEntry],
        all_equilibria: Sequence[PhaseEquilibrium],
        config: OptimizationConfig,
        *,
        console=None,
    ) -> "OptimizationState":
        if console is None:
            console = get_console()

        if not all_equilibria:
            raise ValueError("No equilibria supplied for optimization.")

        all_phases = tuple(all_phases)
        all_equilibria = tuple(all_equilibria)
        phase_names = [phase.phase_name for phase in all_phases]
        duplicate_phase_names = {
            phase_name for phase_name in phase_names if phase_names.count(phase_name) > 1
        }
        if duplicate_phase_names:
            raise ValueError(
                f"Phase names must be unique: {sorted(duplicate_phase_names)}"
            )

        parameter0 = (
            _snapshot_trainable_model_parameters(all_phases)
            if config.regularize_difference
            else None
        )

        mu_checkpoint = (
            Path(config.mu_checkpoint_path)
            if config.mu_checkpoint_path is not None
            else None
        )
        load_mu_checkpoint = mu_checkpoint is not None and mu_checkpoint.exists()

        console.rule("BUILD EQUILIBRIUM LOSSES")
        if load_mu_checkpoint:
            console.print(f"loading initialized mu from {mu_checkpoint}")
        equilibrium_losses = torch.nn.ModuleList(
            [
                SinglePhaseEquilibriumLoss(
                    equilibrium,
                    all_phases,
                    n_samples=config.n_samples,
                    tau=config.tau,
                    relu_margin=config.relu_margin,
                    unstable_huber_beta=config.unstable_huber_beta,
                    n_steps=config.n_steps,
                    delta=config.delta,
                    mu_init_lr=config.mu_init_lr,
                    mu_init_max_iter=config.mu_init_max_iter,
                    mu_convergence_tol=config.mu_convergence_tol,
                    mu_init_cosine_decay=config.mu_init_cosine_decay,
                    mu_strategy=config.mu_strategy,
                    analytic_condition_threshold=config.analytic_condition_threshold,
                    initialize_mu=not load_mu_checkpoint,
                    console=console,
                )
                for equilibrium in all_equilibria
            ]
        )
        if load_mu_checkpoint:
            checkpoint = torch.load(mu_checkpoint, map_location=DEFAULT_DEVICE)
            state_dict = checkpoint.get("equilibrium_losses_state", checkpoint)
            equilibrium_losses.load_state_dict(state_dict)
            console.print(f"loaded initialized mu from {mu_checkpoint}")
        elif mu_checkpoint is not None:
            mu_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "equilibrium_losses_state": equilibrium_losses.state_dict(),
                    "n_equilibria": len(all_equilibria),
                    "mu_strategy": [
                        equilibrium_loss.strategy
                        for equilibrium_loss in equilibrium_losses
                    ],
                },
                mu_checkpoint,
            )
            console.print(f"saved initialized mu to {mu_checkpoint}")

        return cls(
            all_phases=all_phases,
            all_equilibria=all_equilibria,
            config=config,
            equilibrium_losses=equilibrium_losses,
            parameter0=parameter0,
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return _collect_trainable_parameters(self.all_phases, self.equilibrium_losses)

    def build_torch_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.trainable_parameters()
        if not parameters:
            raise ValueError(
                "No trainable parameters found in the supplied phases/losses."
            )
        self.torch_optimizer = self.config.optimizer_cls(parameters, lr=self.config.lr)
        return self.torch_optimizer

    def build_scheduler(self, total_steps: int):
        if not self.config.cosine_decay:
            self.scheduler = None
            return None

        min_lr_factor = float(self.config.min_lr_factor)
        if min_lr_factor < 0.0 or min_lr_factor > 1.0:
            raise ValueError("min_lr_factor must be between 0 and 1.")

        def cosine_lr_factor(step: int) -> float:
            progress = min(max(step, 0), total_steps) / total_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        if self.torch_optimizer is None:
            self.build_torch_optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.torch_optimizer,
            lr_lambda=cosine_lr_factor,
        )
        return self.scheduler

    def loss_parts(self, batch_indices: Sequence[int]) -> dict[str, object]:
        return _aggregate_loss_parts(
            self.equilibrium_losses,
            batch_indices,
            self.all_phases,
            stable_weight=self.config.stable_weight,
            unstable_weight=self.config.unstable_weight,
            regularization_weight=self.config.regularization_weight,
            parameter0=self.parameter0,
        )

    def config_state_dict(self) -> dict[str, object]:
        config = {
            field.name: getattr(self.config, field.name)
            for field in dataclasses.fields(self.config)
            if field.name != "optimizer_cls"
        }
        config["optimizer_cls"] = self.config.optimizer_cls.__name__
        if isinstance(config["mu_checkpoint_path"], Path):
            config["mu_checkpoint_path"] = str(config["mu_checkpoint_path"])
        if isinstance(config["checkpoint_dir"], Path):
            config["checkpoint_dir"] = str(config["checkpoint_dir"])
        if isinstance(config["restart_from"], Path):
            config["restart_from"] = str(config["restart_from"])
        return config

    def state_dict(self) -> dict[str, object]:
        return {
            "phase_models": {
                phase.phase_name: phase.model.state_dict()
                for phase in self.all_phases
            },
            "equilibrium_losses": self.equilibrium_losses.state_dict(),
            "torch_optimizer": (
                None
                if self.torch_optimizer is None
                else self.torch_optimizer.state_dict()
            ),
            "scheduler": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "history": list(self.history),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": self.config_state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        phase_model_states = state_dict.get("phase_models", {})
        for phase in self.all_phases:
            if phase.phase_name in phase_model_states:
                phase.model.load_state_dict(phase_model_states[phase.phase_name])
        if "equilibrium_losses" in state_dict:
            self.equilibrium_losses.load_state_dict(state_dict["equilibrium_losses"])
        if self.torch_optimizer is not None and state_dict.get("torch_optimizer"):
            self.torch_optimizer.load_state_dict(state_dict["torch_optimizer"])
        if self.scheduler is not None and state_dict.get("scheduler"):
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.history = list(state_dict.get("history", []))
        self.epoch = int(state_dict.get("epoch", 0))
        self.global_step = int(state_dict.get("global_step", 0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=DEFAULT_DEVICE)
        self.load_state_dict(state_dict)

    def checkpoint_dir(self) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir)

    def checkpoint_path(self, filename: str) -> Path | None:
        checkpoint_dir = self.checkpoint_dir()
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / filename


def optimize_thermodynamic_parameters(
    all_phases: Sequence[PhaseEntry],
    all_equilibria: Sequence[PhaseEquilibrium],
    config: OptimizationConfig | None = None,
    *,
    restart_from: str | Path | None = None,
    console=None,
):
    """Optimize thermodynamic models using OptimizationConfig/OptimizationState."""
    if console is None:
        console = get_console()

    restart_path = Path(restart_from) if restart_from is not None else None
    if config is None:
        if restart_path is not None:
            checkpoint = torch.load(restart_path, map_location=DEFAULT_DEVICE)
            config = OptimizationConfig.from_state_dict(checkpoint)
        else:
            config = OptimizationConfig()
    if restart_path is not None:
        config.restart_from = restart_path

    state = OptimizationState.initialize(
        all_phases,
        all_equilibria,
        config,
        console=console,
    )

    n_equilibria = len(state.all_equilibria)
    effective_batch_size = (
        n_equilibria if config.batch_size is None else int(config.batch_size)
    )
    effective_batch_size = max(1, min(effective_batch_size, n_equilibria))
    batches_per_epoch = math.ceil(n_equilibria / effective_batch_size)
    total_steps = max(1, int(config.epochs) * batches_per_epoch)

    console.rule("OPTIMIZATION PARAMETERS")
    console.print(f"epochs = {config.epochs}")
    if config.loss_threshold is not None:
        console.print(f"loss threshold = {config.loss_threshold}")
    console.print(f"lr = {config.lr}")
    if config.cosine_decay:
        console.print(
            f"lr schedule = cosine decay to {config.min_lr_factor:g} * lr"
        )
    console.print(f"batch size = {effective_batch_size}")
    console.print(f"sampling density = {config.n_samples}")
    console.print(f"tau = {config.tau}")
    console.print(f"unstable huber beta = {config.unstable_huber_beta}")
    console.print(f"exp-gradient steps = {config.n_steps}")
    console.print(f"exp-gradient delta = {config.delta}")
    console.print(f"stable weight = {config.stable_weight}")
    console.print(f"unstable weight = {config.unstable_weight}")
    console.print(f"regularization weight = {config.regularization_weight}")
    console.print(f"regularize difference = {config.regularize_difference}")
    console.print(f"optimizer = {config.optimizer_cls}")
    average_T = sum(eq.temperature for eq in state.all_equilibria) / n_equilibria
    console.print(f"average temp of equilibria = {average_T}")

    console.rule("MODEL PARAMETERS")
    for phase in state.all_phases:
        n_parameters = sum(
            parameter.numel()
            for parameter in phase.model.parameters()
            if parameter.requires_grad
        )
        console.print(f"{phase.phase_name:<20s} ({n_parameters:d} parameters)")
    latent_mu_parameters = sum(
        parameter.numel()
        for parameter in state.equilibrium_losses.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'latent mu':<20s} ({latent_mu_parameters:d} parameters)")

    all_indices = list(range(n_equilibria))
    optimizer = state.build_torch_optimizer()
    scheduler = state.build_scheduler(total_steps)
    if config.restart_from is not None:
        restart_path = Path(config.restart_from)
        console.print(f"loading optimization state from {restart_path}")
        state.load(restart_path)
        console.print(
            f"loaded optimization state at epoch {state.epoch}, "
            f"step {state.global_step}"
        )

    with torch.no_grad():
        state.initial_loss_parts = state.loss_parts(all_indices)

    best_path = state.checkpoint_path(config.best_filename)
    last_path = state.checkpoint_path(config.last_filename)
    best_loss = float(state.initial_loss_parts["total"].detach().cpu())
    if best_path is not None:
        state.save(best_path)
        console.print(
            f"saved best optimization state to {best_path} "
            f"(loss={best_loss:.2e})"
        )

    console.rule("OPTIMIZE")
    t0 = time.time()
    should_stop = False
    for epoch in range(state.epoch + 1, config.epochs + 1):
        state.epoch = epoch
        if effective_batch_size == n_equilibria:
            batches = [all_indices]
        else:
            shuffled_indices = torch.randperm(n_equilibria).tolist()
            batches = [
                shuffled_indices[start:start + effective_batch_size]
                for start in range(0, len(shuffled_indices), effective_batch_size)
            ]

        for batch_indices in batches:
            state.global_step += 1
            optimizer.zero_grad(set_to_none=True)

            loss_parts = state.loss_parts(batch_indices)
            total_loss = loss_parts["total"]
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            state.history.append(float(total_loss.detach().cpu()))

            if (
                config.print_every
                and (
                    state.global_step == 1
                    or state.global_step % config.print_every == 0
                )
            ):
                stable_loss = float(loss_parts["stable"].detach().cpu())
                unstable_loss = float(loss_parts["unstable"].detach().cpu())
                regularization_loss = float(
                    loss_parts["regularization"].detach().cpu()
                )
                current_lr = optimizer.param_groups[0]["lr"]
                t1 = time.time()
                console.print(
                    f"epoch {epoch:>4d}/{config.epochs}, "
                    f"step {state.global_step:>6d}: "
                    f"lr={current_lr:10.2e}, "
                    f"loss={state.history[-1]:10.2e}, "
                    f"stable={stable_loss:10.2e}, "
                    f"unstable={unstable_loss:10.2e}, "
                    f"regularization={regularization_loss:10.2e}, "
                    f'time={t1-t0:>.3f} sec.'
                )
                t0 = t1

            if (
                config.loss_threshold is not None
                and state.history[-1] <= config.loss_threshold
            ):
                if config.print_every:
                    console.print(
                        "\n"
                        f"stopping early at epoch {epoch}/{config.epochs}, "
                        f"step {state.global_step}: "
                        f"loss={state.history[-1]:.2e} <= "
                        f"threshold={config.loss_threshold:.2e}"
                    )
                should_stop = True
                break
        if should_stop:
            break
        if (
            last_path is not None
            and config.save_every is not None
            and config.save_every > 0
            and epoch % config.save_every == 0
        ):
            state.save(last_path)
            console.print(f"saved latest optimization state to {last_path}")

        with torch.no_grad():
            epoch_loss = float(state.loss_parts(all_indices)["total"].detach().cpu())
        if best_path is not None and epoch_loss < best_loss:
            best_loss = epoch_loss
            state.save(best_path)
            console.print(
                f"saved best optimization state to {best_path} "
                f"(loss={best_loss:.2e})"
            )

    with torch.no_grad():
        state.final_loss_parts = state.loss_parts(all_indices)
    final_loss = float(state.final_loss_parts["total"].detach().cpu())
    if last_path is not None:
        state.save(last_path)
        console.print(f"saved latest optimization state to {last_path}")
    if best_path is not None and final_loss < best_loss:
        best_loss = final_loss
        state.save(best_path)
        console.print(
            f"saved best optimization state to {best_path} "
            f"(loss={best_loss:.2e})"
        )

    if state.history:
        console.print(
            "initial loss: "
            f"{float(state.initial_loss_parts['total'].detach().cpu()):.2e}"
        )
        console.print(f"final   loss: {final_loss:.2e}")
    if config.print_final_results:
        console.rule("PHI AT EQUILIBRIA (FINAL)")
        _print_phi_at_equilibria(
            state.equilibrium_losses,
            state.final_loss_parts,
            console=console,
        )
    console.rule("FINISHED")
    return state.history


def optimize_thermodynamic_parameters_old(
    all_phases: Sequence[PhaseEntry],
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
    stable_weight: float = 1.0,
    unstable_weight: float = 1.0,
    regularization_weight: float = 1.0e-12,
    regularize_difference: bool = False,
    n_samples: int = 64,
    tau: float | None = None,
    relu_margin: float = 0.0,
    unstable_huber_beta: float | None = 1.0,
    n_steps: int = 6,
    delta: float = 0.3,
    mu_init_lr: float = 5000.0,
    mu_init_max_iter: int = 1000,
    mu_convergence_tol: float = 10.0,
    mu_init_cosine_decay: bool = True,
    mu_strategy: str = "auto",
    analytic_condition_threshold: float = 1.0e10,
    console=None,
    print_final_results: bool = True,
    mu_checkpoint_path: str | Path | None = None,
):
    """Optimize thermodynamic models using per-equilibrium loss objects."""
    if console is None:
        console = get_console()

    if not all_equilibria:
        raise ValueError("No equilibria supplied for optimization.")

    all_phases = tuple(all_phases)
    phase_names = [phase.phase_name for phase in all_phases]
    duplicate_phase_names = {
        phase_name for phase_name in phase_names if phase_names.count(phase_name) > 1
    }
    if duplicate_phase_names:
        raise ValueError(f"Phase names must be unique: {sorted(duplicate_phase_names)}")

    parameter0 = (
        _snapshot_trainable_model_parameters(all_phases)
        if regularize_difference
        else None
    )

    mu_checkpoint = Path(mu_checkpoint_path) if mu_checkpoint_path is not None else None
    load_mu_checkpoint = mu_checkpoint is not None and mu_checkpoint.exists()

    console.rule("BUILD EQUILIBRIUM LOSSES")
    if load_mu_checkpoint:
        console.print(f"loading initialized mu from {mu_checkpoint}")
    equilibrium_losses = torch.nn.ModuleList(
        [
            SinglePhaseEquilibriumLoss(
                equilibrium,
                all_phases,
                n_samples=n_samples,
                tau=tau,
                relu_margin=relu_margin,
                unstable_huber_beta=unstable_huber_beta,
                n_steps=n_steps,
                delta=delta,
                mu_init_lr=mu_init_lr,
                mu_init_max_iter=mu_init_max_iter,
                mu_convergence_tol=mu_convergence_tol,
                mu_init_cosine_decay=mu_init_cosine_decay,
                mu_strategy=mu_strategy,
                analytic_condition_threshold=analytic_condition_threshold,
                initialize_mu=not load_mu_checkpoint,
                console=console,
            )
            for equilibrium in all_equilibria
        ]
    )
    if load_mu_checkpoint:
        checkpoint = torch.load(mu_checkpoint, map_location=DEFAULT_DEVICE)
        state_dict = checkpoint.get("equilibrium_losses_state", checkpoint)
        equilibrium_losses.load_state_dict(state_dict)
        console.print(f"loaded initialized mu from {mu_checkpoint}")
    elif mu_checkpoint is not None:
        mu_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "equilibrium_losses_state": equilibrium_losses.state_dict(),
                "n_equilibria": len(all_equilibria),
                "mu_strategy": [
                    equilibrium_loss.strategy
                    for equilibrium_loss in equilibrium_losses
                ],
            },
            mu_checkpoint,
        )
        console.print(f"saved initialized mu to {mu_checkpoint}")

    parameters = _collect_trainable_parameters(all_phases, equilibrium_losses)
    if not parameters:
        raise ValueError("No trainable parameters found in the supplied phases/losses.")

    if batch_size is None:
        batch_size = len(all_equilibria)
    batch_size = max(1, min(int(batch_size), len(all_equilibria)))
    batches_per_epoch = math.ceil(len(all_equilibria) / batch_size)
    total_steps = max(1, int(epochs) * batches_per_epoch)

    console.rule("OPTIMIZATION PARAMETERS")
    console.print(f"epochs = {epochs}")
    if loss_threshold is not None:
        console.print(f"loss threshold = {loss_threshold}")
    console.print(f"lr = {lr}")
    if cosine_decay:
        console.print(f"lr schedule = cosine decay to {min_lr_factor:g} * lr")
    console.print(f"batch size = {batch_size}")
    console.print(f"sampling density = {n_samples}")
    console.print(f"tau = {tau}")
    console.print(f"unstable huber beta = {unstable_huber_beta}")
    console.print(f"exp-gradient steps = {n_steps}")
    console.print(f"exp-gradient delta = {delta}")
    console.print(f"stable weight = {stable_weight}")
    console.print(f"unstable weight = {unstable_weight}")
    console.print(f"regularization weight = {regularization_weight}")
    console.print(f"regularize difference = {regularize_difference}")
    console.print(f"optimizer = {optimizer_cls}")
    average_T = sum(eq.temperature for eq in all_equilibria) / len(all_equilibria)
    console.print(f"average temp of equilibria = {average_T}")

    console.rule("MODEL PARAMETERS")
    for phase in all_phases:
        n_parameters = sum(
            parameter.numel()
            for parameter in phase.model.parameters()
            if parameter.requires_grad
        )
        console.print(f"{phase.phase_name:<20s} ({n_parameters:d} parameters)")
    latent_mu_parameters = sum(
        parameter.numel()
        for parameter in equilibrium_losses.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'latent mu':<20s} ({latent_mu_parameters:d} parameters)")

    #console.rule("EQUILIBRIA")
    #for index, equilibrium_loss in enumerate(equilibrium_losses):
    #    console.print(
    #        f"{index:3d}) [{equilibrium_loss.strategy}] "
    #        f"{equilibrium_loss.equilibrium}"
    #    )

    all_indices = list(range(len(all_equilibria)))
    with torch.no_grad():
        initial_loss_parts = _aggregate_loss_parts(
            equilibrium_losses,
            all_indices,
            all_phases,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
            parameter0=parameter0,
        )
    #console.rule("PHI AT EQUILIBRIA (INITIAL)")
    #_print_phi_at_equilibria(
    #    equilibrium_losses,
    #    initial_loss_parts,
    #    console=console,
    #)

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
    console.rule("OPTIMIZE")
    global_step = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if batch_size == len(all_equilibria):
            batches = [all_indices]
        else:
            shuffled_indices = torch.randperm(len(all_equilibria)).tolist()
            batches = [
                shuffled_indices[start:start + batch_size]
                for start in range(0, len(shuffled_indices), batch_size)
            ]

        for batch_indices in batches:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            loss_parts = _aggregate_loss_parts(
                equilibrium_losses,
                batch_indices,
                all_phases,
                stable_weight=stable_weight,
                unstable_weight=unstable_weight,
                regularization_weight=regularization_weight,
                parameter0=parameter0,
            )
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
                t1 = time.time()
                console.print(
                    f"epoch {epoch:>4d}/{epochs}, "
                    f"step {global_step:>6d}: "
                    f"lr={current_lr:10.2e}, "
                    f"loss={history[-1]:10.2e}, "
                    f"stable={stable_loss:10.2e}, "
                    f"unstable={unstable_loss:10.2e}, "
                    f"regularization={regularization_loss:10.2e}, "
                    f'time={t1-t0:>.3f} sec.'
                )
                t0 = t1

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
        final_loss_parts = _aggregate_loss_parts(
            equilibrium_losses,
            all_indices,
            all_phases,
            stable_weight=stable_weight,
            unstable_weight=unstable_weight,
            regularization_weight=regularization_weight,
            parameter0=parameter0,
        )
    final_loss = float(final_loss_parts["total"].detach().cpu())

    if history:
        console.print(
            f"initial loss: {float(initial_loss_parts['total'].detach().cpu()):.2e}"
        )
        console.print(f"final   loss: {final_loss:.2e}")
    if print_final_results:
        console.rule("PHI AT EQUILIBRIA (FINAL)")
        _print_phi_at_equilibria(
            equilibrium_losses,
            final_loss_parts,
            console=console,
        )
    console.rule("FINISHED")
    return history
