import dataclasses
from typing import Sequence
import itertools
import math
from pathlib import Path
import time

import torch
from rich.console import Console

from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .loss_function import SinglePhaseEquilibriumLoss
from .phase import PhaseEquilibrium
from .models import ThermodynamicSystem

def get_console():
    return Console()


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Disable optimization of all parameters in a torch model."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _snapshot_trainable_model_parameters(
    system: ThermodynamicSystem,
) -> dict[str, torch.Tensor]:
    return {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in system.named_parameters()
        if parameter.requires_grad
    }


def _regularization_loss(
    system: ThermodynamicSystem,
    parameter0: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    total = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    if parameter0 is None:
        for parameter in system.parameters():
            if parameter.requires_grad:
                total = total + parameter.square().sum()
        return total

    for parameter_name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_name not in parameter0:
            raise ValueError(f"Missing parameter reference for {parameter_name}.")
        reference_parameter = parameter0[parameter_name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if reference_parameter.shape != parameter.shape:
            raise ValueError(
                f"Reference shape mismatch for {parameter_name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(reference_parameter.shape)}."
            )
        total = total + (parameter - reference_parameter).square().sum()
    return total


def _collect_trainable_parameters(
    system: ThermodynamicSystem,
    equilibrium_losses: torch.nn.ModuleList,
) -> list[torch.nn.Parameter]:
    parameters = []
    seen_ids = set()
    for parameter in itertools.chain(system.parameters(), equilibrium_losses.parameters()):
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_ids:
            continue
        seen_ids.add(parameter_id)
        parameters.append(parameter)
    return parameters


def _aggregate_loss_parts(
    equilibrium_losses: Sequence[SinglePhaseEquilibriumLoss],
    batch_indices: Sequence[int],
    system: ThermodynamicSystem,
    *,
    stable_weight: float,
    unstable_weight: float,
    regularization_weight: float,
    parameter0: dict[str, torch.Tensor] | None,
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
            regularization_weight * _regularization_loss(system, parameter0)
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
    epochs: int = 500
    lr: float = 1000.0
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam
    loss_threshold: float | None = None
    cosine_decay: bool = True
    min_lr_factor: float = 0.1

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

    mu_convergence_tol: float = 50.0
    mu_init_lr: float = 5000.0
    mu_init_max_iter: int = 1000
    mu_init_cosine_decay: bool = True
    mu_strategy: str = "auto"


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
    system: ThermodynamicSystem
    equilibria: tuple[PhaseEquilibrium, ...]
    config: OptimizationConfig
    equilibrium_losses: torch.nn.ModuleList
    parameter0: dict[str, torch.Tensor] | None = None
    torch_optimizer: torch.optim.Optimizer | None = None
    scheduler: object | None = None
    history: list[float] = dataclasses.field(default_factory=list)
    epoch: int = 0
    global_step: int = 0
    best_loss: float = math.inf
    initial_loss_parts: dict[str, object] | None = None
    final_loss_parts: dict[str, object] | None = None

    @classmethod
    def initialize(
        cls,
        system: ThermodynamicSystem,
        equilibria: Sequence[PhaseEquilibrium],
        config: OptimizationConfig,
        *,
        initialize_mu: bool = True,
        console=None,
    ) -> "OptimizationState":
        if console is None:
            console = get_console()

        if not equilibria:
            raise ValueError("No equilibria supplied for optimization.")

        equilibria = tuple(equilibria)

        parameter0 = (
            _snapshot_trainable_model_parameters(system)
            if config.regularize_difference
            else None
        )

        console.rule("BUILD EQUILIBRIUM LOSSES")
        equilibrium_losses = torch.nn.ModuleList(
            [
                SinglePhaseEquilibriumLoss(
                    equilibrium,
                    system,
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
                    initialize_mu=initialize_mu,
                    console=console,
                )
                for equilibrium in equilibria
            ]
        )

        return cls(
            system=system,
            equilibria=equilibria,
            config=config,
            equilibrium_losses=equilibrium_losses,
            parameter0=parameter0,
        )


    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return _collect_trainable_parameters(self.system, self.equilibrium_losses)


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
            self.system,
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
        return config


    def state_dict(self) -> dict[str, object]:
        return {
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
            "best_loss": self.best_loss,
            "parameter0": self.parameter0,
            "config": self.config_state_dict(),
        }


    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if "equilibrium_losses" in state_dict:
            self.equilibrium_losses.load_state_dict(state_dict["equilibrium_losses"])
        if self.torch_optimizer is not None and state_dict.get("torch_optimizer"):
            self.torch_optimizer.load_state_dict(state_dict["torch_optimizer"])
        if self.scheduler is not None and state_dict.get("scheduler"):
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.history = list(state_dict.get("history", []))
        self.epoch = int(state_dict.get("epoch", 0))
        self.global_step = int(state_dict.get("global_step", 0))
        self.best_loss = float(state_dict.get("best_loss", math.inf))
        self.parameter0 = state_dict.get("parameter0", self.parameter0)


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=DEFAULT_DEVICE)
        self.load_state_dict(state_dict)


def optimize_thermodynamic_parameters(
    system: ThermodynamicSystem | None = None,
    equilibria: Sequence[PhaseEquilibrium] | None = None,
    config: OptimizationConfig | None = None,
    *,
    checkpoint_dir: str | Path | None = 'checkpoint',
    restart: bool = False,
    save_every: int | None = 50,
    best_filename: str = "best.pt",
    last_filename: str = "last.pt",
    opt_state_filename: str = "opt_state.pt",
    data_filename: str = "data.pt",
    config_filename: str = "config.pt",
    print_every: int = 10,
    print_final_results: bool = True,
    console=None,
):
    """Optimize thermodynamic models using OptimizationConfig/OptimizationState."""
    if console is None:
        console = get_console()

    if checkpoint_dir is None:
        checkpoint_path = None
        last_model_path = None
        best_model_path = None
        opt_state_path = None
        data_path = None
        config_path = None
    else:
        checkpoint_path = Path(checkpoint_dir)
        last_model_path = checkpoint_path / last_filename
        best_model_path = checkpoint_path / best_filename
        opt_state_path = checkpoint_path / opt_state_filename
        data_path = checkpoint_path / data_filename
        config_path = checkpoint_path / config_filename

    if restart:
        if checkpoint_path is None:
            raise ValueError("checkpoint_dir is required when restart=True.")
        console.print(f"restarting optimization from {checkpoint_path}")
        console.print(f'optimization inputs are ignored')
        system = torch.load(
            last_model_path,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        equilibria = torch.load(
            data_path,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        config = torch.load(
            config_path,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        if not isinstance(system, ThermodynamicSystem):
            raise TypeError(
                f"Expected {last_model_path} to contain a ThermodynamicSystem, "
                f"got {type(system).__name__}."
            )
        if not isinstance(config, OptimizationConfig):
            raise TypeError(
                f"Expected {config_path} to contain an OptimizationConfig, "
                f"got {type(config).__name__}."
            )
    else:
        if system is None:
            raise ValueError("system is required when restart=False.")
        if equilibria is None:
            raise ValueError("equilibria is required when restart=False.")
        if config is None:
            raise ValueError("config is required when restart=False.")

    state = OptimizationState.initialize(
        system,
        equilibria,
        config,
        initialize_mu=not restart,
        console=console,
    )

    if restart:
        # load state
        state.build_torch_optimizer()
        n_equilibria_for_steps = len(state.equilibria)
        effective_batch_size_for_steps = (
            n_equilibria_for_steps
            if config.batch_size is None
            else int(config.batch_size)
        )
        effective_batch_size_for_steps = max(
            1,
            min(effective_batch_size_for_steps, n_equilibria_for_steps),
        )
        batches_per_epoch_for_steps = math.ceil(
            n_equilibria_for_steps / effective_batch_size_for_steps
        )
        state.build_scheduler(max(1, int(config.epochs) * batches_per_epoch_for_steps))
        state.load(opt_state_path)
        console.print(
            f"loaded optimization state at epoch {state.epoch}, "
            f"step {state.global_step}"
        )

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if not restart:
            torch.save(tuple(state.equilibria), data_path)
            torch.save(config, config_path)

    # proceed optimization
    n_equilibria = len(state.equilibria)
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
    average_T = sum(eq.temperature for eq in state.equilibria) / n_equilibria
    console.print(f"average temp of equilibria = {average_T}")

    console.rule("MODEL PARAMETERS")
    trainable_model_parameters = sum(
        parameter.numel()
        for parameter in state.system.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'thermodynamic system':<24s} ({trainable_model_parameters:d} parameters)")
    latent_mu_parameters = sum(
        parameter.numel()
        for parameter in state.equilibrium_losses.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'latent mu':<24s} ({latent_mu_parameters:d} parameters)")

    all_indices = list(range(n_equilibria))
    optimizer = state.torch_optimizer or state.build_torch_optimizer()
    scheduler = state.scheduler or state.build_scheduler(total_steps)

    with torch.no_grad():
        state.initial_loss_parts = state.loss_parts(all_indices)

    if not math.isfinite(state.best_loss):
        state.best_loss = float(state.initial_loss_parts["total"].detach().cpu())
    if checkpoint_path is not None and not restart:
        torch.save(state.system, best_model_path)
        torch.save(state.system, last_model_path)
        state.save(opt_state_path)
        console.print(
            f"saved initial checkpoint to {checkpoint_path} "
            f"(loss={state.best_loss:.2e})"
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
                print_every
                and (
                    state.global_step == 1
                    or state.global_step % print_every == 0
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
                if print_every:
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
            checkpoint_path is not None
            and save_every is not None
            and save_every > 0
            and epoch % save_every == 0
        ):
            torch.save(state.system, last_model_path)
            state.save(opt_state_path)

        with torch.no_grad():
            epoch_loss = float(state.loss_parts(all_indices)["total"].detach().cpu())
        
        if checkpoint_path is not None and epoch_loss < state.best_loss:
            state.best_loss = epoch_loss
            torch.save(state.system, best_model_path)
            state.save(opt_state_path)

    with torch.no_grad():
        state.final_loss_parts = state.loss_parts(all_indices)
    final_loss = float(state.final_loss_parts["total"].detach().cpu())
    
    if checkpoint_path is not None:
        torch.save(state.system, last_model_path)
        state.save(opt_state_path)
        console.print(f"saved latest model to {last_model_path}")
        console.print(f"saved latest optimization state to {opt_state_path}")
    
    if checkpoint_path is not None and final_loss < state.best_loss:
        state.best_loss = final_loss
        torch.save(state.system, best_model_path)
        state.save(opt_state_path)
        console.print(
            f"saved best model to {best_model_path} "
            f"(loss={state.best_loss:.2e})"
        )

    if state.history:
        console.print(
            "initial loss: "
            f"{float(state.initial_loss_parts['total'].detach().cpu()):.2e}"
        )
        console.print(f"final   loss: {final_loss:.2e}")
        
    if print_final_results:
        console.rule("PHI AT EQUILIBRIA (FINAL)")
        _print_phi_at_equilibria(
            state.equilibrium_losses,
            state.final_loss_parts,
            console=console,
        )
    console.rule("FINISHED")
    return state.history
