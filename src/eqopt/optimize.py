import dataclasses
from typing import Sequence
import math
from pathlib import Path
import time

import torch
from rich.console import Console

from .config import OptimizationConfig
from .dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .loss_function import (
    EquilibriumLossRecord,
    PhaseEquilibriumOptState,
    phase_equilibrium_loss_parts,
    print_phi_at_equilibria,
)
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


def _deduplicate_trainable_parameters(
    parameters_iterable,
    seen_ids: set[int] | None = None,
) -> list[torch.nn.Parameter]:
    """collect all unique training parameters from a list"""
    parameters = []
    if seen_ids is None:
        seen_ids = set()
    for parameter in parameters_iterable:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_ids:
            continue
        seen_ids.add(parameter_id)
        parameters.append(parameter)
    return parameters


def _collect_trainable_model_parameters(
    system: ThermodynamicSystem,
) -> list[torch.nn.Parameter]:
    return _deduplicate_trainable_parameters(system.parameters())


def _collect_trainable_latent_mu_parameters(
    equilibrium_losses: torch.nn.ModuleList,
) -> list[torch.nn.Parameter]:
    return _deduplicate_trainable_parameters(equilibrium_losses.parameters())


def _save_equilibrium_states_without_runtime_data(
    equilibrium_states: Sequence[PhaseEquilibriumOptState],
    path: str | Path,
) -> None:
    """
    runtime data are caches managed by the thermodynamic models.

    when saving the equilibrium state, they are not saved by setting
    them to None
    """
    runtime_data_by_index = [
        getattr(eq_state, "runtime_data", None)
        for eq_state in equilibrium_states
    ]
    try:
        for eq_state in equilibrium_states:
            eq_state.clear_runtime_data()
        torch.save(equilibrium_states, path)
    finally:
        for eq_state, runtime_data in zip(
            equilibrium_states,
            runtime_data_by_index,
            strict=True,
        ):
            eq_state.runtime_data = runtime_data


def _aggregate_loss_parts(
    equilibrium_losses: Sequence[PhaseEquilibriumOptState],
    batch_indices: Sequence[int],
    system: ThermodynamicSystem,
    *,
    config: OptimizationConfig,
    stable_weight: float,
    unstable_weight: float,
    regularization_weight: float,
    parameter0: dict[str, torch.Tensor] | None,
) -> dict[str, object]:
    stable = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    unstable = torch.zeros((), device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
    phi_at_equilibria: list[tuple[int, EquilibriumLossRecord]] = []

    for equilibrium_index in batch_indices:
        record = phase_equilibrium_loss_parts(
            equilibrium_losses[equilibrium_index],
            system,
            relu_margin=config.relu_margin,
            unstable_huber_beta=config.unstable_huber_beta,
            use_huber_for_stable_phases=config.use_huber_for_stable_phases,
            scale_energy_by_rt=config.scale_energy_by_rt,
        )
        stable = stable + record.stable_loss
        unstable = unstable + record.unstable_loss
        phi_at_equilibria.append((equilibrium_index, record))

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
    loss_parts: dict[str, object],
    *,
    console,
) -> None:
    for equilibrium_index, record in loss_parts["phi_at_equilibria"]:
        console.print(f"{equilibrium_index:3d}) ", end="")
        print_phi_at_equilibria(record, console=console)


def _empty_history() -> dict[str, list[float | int]]:
    return {
        "iepoch": [],
        "stable_loss": [],
        "unstable_loss": [],
        "regularization_loss": [],
        "weighted_total_loss": [],
    }


def _normalize_history(history: object) -> dict[str, list[float | int]]:
    normalized = _empty_history()
    if isinstance(history, dict):
        for key in normalized:
            normalized[key] = list(history.get(key, []))
        return normalized

    # Backward compatibility for older checkpoints with a flat total-loss list.
    if isinstance(history, list):
        normalized["iepoch"] = list(range(1, len(history) + 1))
        normalized["weighted_total_loss"] = list(history)
    return normalized


@dataclasses.dataclass
class OptimizationState:
    config: OptimizationConfig | None = None
    parameter0: dict[str, torch.Tensor] | None = None
    model_optimizer_state: dict[str, object] | None = None
    mu_optimizer_state: dict[str, object] | None = None
    model_scheduler_state: dict[str, object] | None = None
    mu_scheduler_state: dict[str, object] | None = None
    history: dict[str, list[float | int]] = dataclasses.field(
        default_factory=_empty_history
    )
    epoch: int = 0
    best_loss: float = math.inf
    initial_loss_parts: dict[str, object] | None = None
    final_loss_parts: dict[str, object] | None = None

    @classmethod
    def create(
        cls,
        system: ThermodynamicSystem,
        config: OptimizationConfig,
    ) -> "OptimizationState":
        return cls(
            config=config,
            parameter0=(
                _snapshot_trainable_model_parameters(system)
                if config.regularize_difference
                else None
            )
        )


    def state_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "model_optimizer_state": self.model_optimizer_state,
            "mu_optimizer_state": self.mu_optimizer_state,
            "model_scheduler_state": self.model_scheduler_state,
            "mu_scheduler_state": self.mu_scheduler_state,
            "history": {
                key: list(value)
                for key, value in self.history.items()
            },
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "parameter0": self.parameter0,
            "initial_loss_parts": self.initial_loss_parts,
            "final_loss_parts": self.final_loss_parts,
        }


    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        config = state_dict.get("config", self.config)
        if isinstance(config, dict):
            config = OptimizationConfig.from_state_dict({"config": config})
        self.config = config
        self.model_optimizer_state = state_dict.get("model_optimizer_state")
        self.mu_optimizer_state = state_dict.get("mu_optimizer_state")
        self.model_scheduler_state = state_dict.get("model_scheduler_state")
        self.mu_scheduler_state = state_dict.get("mu_scheduler_state")
        self.history = _normalize_history(state_dict.get("history", {}))
        self.epoch = int(state_dict.get("epoch", 0))
        self.best_loss = float(state_dict.get("best_loss", math.inf))
        self.parameter0 = state_dict.get("parameter0", self.parameter0)
        self.initial_loss_parts = state_dict.get(
            "initial_loss_parts",
            self.initial_loss_parts,
        )
        self.final_loss_parts = state_dict.get(
            "final_loss_parts",
            self.final_loss_parts,
        )


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


    def load(self, path: str | Path) -> None:
        state_dict = torch.load(
            path,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        self.load_state_dict(state_dict)


    def update_from_runtime(
        self,
        model_optimizer: torch.optim.Optimizer | None,
        mu_optimizer: torch.optim.Optimizer | None,
        model_scheduler,
        mu_scheduler,
    ) -> None:
        self.model_optimizer_state = (
            None if model_optimizer is None else model_optimizer.state_dict()
        )
        self.mu_optimizer_state = (
            None if mu_optimizer is None else mu_optimizer.state_dict()
        )
        self.model_scheduler_state = (
            None if model_scheduler is None else model_scheduler.state_dict()
        )
        self.mu_scheduler_state = (
            None if mu_scheduler is None else mu_scheduler.state_dict()
        )


    def record_history(
        self,
        iepoch: int,
        loss_parts: dict[str, object],
    ) -> None:
        self.history["iepoch"].append(int(iepoch))
        self.history["stable_loss"].append(
            float(loss_parts["stable"].detach().cpu())
        )
        self.history["unstable_loss"].append(
            float(loss_parts["unstable"].detach().cpu())
        )
        self.history["regularization_loss"].append(
            float(loss_parts["regularization"].detach().cpu())
        )
        self.history["weighted_total_loss"].append(
            float(loss_parts["total"].detach().cpu())
        )


    @classmethod
    def from_file(cls, path: str | Path) -> "OptimizationState":
        loaded = torch.load(
            path,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
        if isinstance(loaded, cls):
            return loaded
        state = cls()
        state.load_state_dict(loaded)
        return state


def _build_model_optimizer(
    system: ThermodynamicSystem,
    config: OptimizationConfig,
) -> torch.optim.Optimizer | None:
    parameters = _collect_trainable_model_parameters(system)
    if not parameters:
        return None
    return config.optimizer_cls(parameters, lr=config.lr)


def _build_mu_optimizer(
    equilibrium_states: torch.nn.ModuleList,
    config: OptimizationConfig,
) -> torch.optim.Optimizer | None:
    parameters = _collect_trainable_latent_mu_parameters(equilibrium_states)
    if not parameters:
        return None
    lr = config.lr if config.latent_mu_lr is None else config.latent_mu_lr
    return config.optimizer_cls(parameters, lr=lr)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    total_steps: int,
):
    if not config.cosine_decay:
        return None

    min_lr_factor = float(config.min_lr_factor)
    if min_lr_factor < 0.0 or min_lr_factor > 1.0:
        raise ValueError("min_lr_factor must be between 0 and 1.")

    def cosine_lr_factor(step: int) -> float:
        progress = min(max(step, 0), total_steps) / total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosine_lr_factor,
    )


def optimize_thermodynamic_parameters(
    system: ThermodynamicSystem | str | Path,
    config: OptimizationConfig | None,
    equilibria: Sequence[PhaseEquilibrium] | None = None,
    equilibrium_states: Sequence[PhaseEquilibriumOptState] | str | Path | None = None,
    optimization_state: OptimizationState | str | Path | None = None,
    *,
    checkpoint_dir: str | Path | None = 'checkpoint',
    record_every: int = 10,
    print_final_results: bool = True,
    console=None,
) -> tuple[ThermodynamicSystem, Sequence[PhaseEquilibriumOptState], OptimizationState]:
    """Optimize thermodynamic models using OptimizationConfig/OptimizationState."""
    if console is None:
        console = get_console()

    if checkpoint_dir is None:
        checkpoint_path = None
        last_model_path = None
        last_eqstate_path = None
        best_model_path = None
        best_eqstate_path = None
        opt_state_path = None
    else:
        checkpoint_path = Path(checkpoint_dir)
        last_model_path = checkpoint_path / 'model_last.pt'
        last_eqstate_path = checkpoint_path / 'equilibria_last.pt'
        best_model_path = checkpoint_path / 'model_best.pt'
        best_eqstate_path = checkpoint_path / 'equilibria_best.pt'
        opt_state_path = checkpoint_path / 'opt_state.pt'

    if isinstance(system, (str, Path)):
        system = torch.load(
            system,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )

    using_existing_state = optimization_state is not None
    if isinstance(optimization_state, (str, Path)):
        state = OptimizationState.from_file(optimization_state)
    elif optimization_state is None:
        if config is None:
            raise ValueError(
                "Either config or optimization_state must be supplied."
            )
        state = OptimizationState.create(system, config)
    else:
        state = optimization_state

    if state.config is None:
        if config is None:
            raise ValueError(
                "OptimizationState does not contain an OptimizationConfig."
            )
        state.config = config
    elif using_existing_state and config is not None:
        console.print(
            "using config stored in optimization state; "
            "input config is ignored"
        )
    config = state.config

    if isinstance(equilibrium_states, (str, Path)):
        equilibrium_states = torch.load(
            equilibrium_states,
            map_location=DEFAULT_DEVICE,
            weights_only=False,
        )
    elif equilibrium_states is None and equilibria is not None:
        console.print(
            'Creating optimization state of phase equilibrium, projecting compositions...',
            end=''
        )
        # we build phase equilibria again with projected composition
        equilibria = tuple(
            PhaseEquilibrium(
                phases=equilibrium.phases,
                phase_compositions=tuple(
                    (
                        None
                        if composition is None
                        else system.project_composition(
                            phase, composition, tol=config.composition_projection_tol)
                    )
                    for phase, composition in zip(
                        equilibrium.phases,
                        equilibrium.phase_compositions,
                        strict=True,
                    )
                ),
                temperature=equilibrium.temperature,
            )
            for equilibrium in equilibria
        )

        console.print('DONE!')
        equilibrium_states = tuple(
            PhaseEquilibriumOptState(eq, mu_strategy=config.mu_strategy)
            for eq in equilibria
        )
        console.rule('we initialize states of equilibria with auxiliary chemical potential')
        system.prepare_for_loss()
        for eq_state in equilibrium_states:
            eq_state.initial_mu_by_minimization(
                system,
                lr=config.mu_init_lr,
                max_iter=config.mu_init_max_iter,
                cosine_decay=config.mu_init_cosine_decay,
                convergence_tol=config.mu_convergence_tol,
                relu_margin=config.relu_margin,
                unstable_huber_beta=config.unstable_huber_beta,
                use_huber_for_stable_phases=config.use_huber_for_stable_phases,
                scale_energy_by_rt=config.scale_energy_by_rt,
                console=console
            )
    elif equilibrium_states is None:
        raise ValueError(
            "Either equilibria or equilibrium_states must be supplied."
        )

    if not isinstance(equilibrium_states, torch.nn.ModuleList):
        equilibrium_states = torch.nn.ModuleList(list(equilibrium_states))

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    # proceed optimization
    n_equilibria = len(equilibrium_states)
    if n_equilibria == 0:
        raise ValueError("At least one equilibrium state is required.")

    effective_batch_size = (
        n_equilibria if config.batch_size is None else int(config.batch_size)
    )
    effective_batch_size = max(1, min(effective_batch_size, n_equilibria))
    batches_per_epoch = math.ceil(n_equilibria / effective_batch_size)
    total_steps = max(1, int(config.epochs) * batches_per_epoch)
    if record_every is None or record_every <= 0:
        raise ValueError("record_every must be a positive integer.")

    console.rule("OPTIMIZATION PARAMETERS")
    console.print(f"epochs = {config.epochs}")
    if config.loss_threshold is not None:
        console.print(f"loss threshold = {config.loss_threshold}")
    console.print(f"model lr = {config.lr}")
    console.print(
        "latent mu lr = "
        f"{config.lr if config.latent_mu_lr is None else config.latent_mu_lr}"
    )
    console.print("optimization mode = joint update")
    if config.cosine_decay:
        console.print(
            f"lr schedule = cosine decay to {config.min_lr_factor:g} * lr"
        )
    console.print(f"batch size = {effective_batch_size}")
    console.print(f"scale energy by RT = {config.scale_energy_by_rt}")
    console.print(f"unstable huber beta = {config.unstable_huber_beta}")
    console.print(
        f"use huber for stable phases = {config.use_huber_for_stable_phases}"
    )
    console.print(f"stable weight = {config.stable_weight}")
    console.print(f"unstable weight = {config.unstable_weight}")
    console.print(f"regularization weight = {config.regularization_weight}")
    console.print(f"regularize difference = {config.regularize_difference}")
    console.print(f"optimizer = {config.optimizer_cls}")
    console.print(f"record every = {record_every} epoch(s)")
    average_T = (
        sum(eq_state.equilibrium.temperature for eq_state in equilibrium_states)
        / n_equilibria
    )
    console.print(f"average temp of equilibria = {average_T}")

    console.rule("MODEL PARAMETERS")
    trainable_model_parameters = sum(
        parameter.numel()
        for parameter in system.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'thermodynamic system':<24s} ({trainable_model_parameters:d} parameters)")
    latent_mu_parameters = sum(
        parameter.numel()
        for parameter in equilibrium_states.parameters()
        if parameter.requires_grad
    )
    console.print(f"{'latent mu':<24s} ({latent_mu_parameters:d} parameters)")

    model_optimizer = _build_model_optimizer(system, config)
    mu_optimizer = _build_mu_optimizer(equilibrium_states, config)
    if model_optimizer is None and mu_optimizer is None:
        raise ValueError(
            "No trainable parameters found in the supplied system/equilibria."
        )

    if model_optimizer is not None and state.model_optimizer_state is not None:
        model_optimizer.load_state_dict(state.model_optimizer_state)
        console.print(
            f"loaded model optimizer state at epoch {state.epoch}"
        )
    if mu_optimizer is not None and state.mu_optimizer_state is not None:
        mu_optimizer.load_state_dict(state.mu_optimizer_state)
        console.print(
            f"loaded latent mu optimizer state at epoch {state.epoch}"
        )

    model_scheduler = (
        None
        if model_optimizer is None
        else _build_scheduler(model_optimizer, config, total_steps)
    )
    mu_scheduler = (
        None
        if mu_optimizer is None
        else _build_scheduler(mu_optimizer, config, total_steps)
    )
    if model_scheduler is not None and state.model_scheduler_state is not None:
        model_scheduler.load_state_dict(state.model_scheduler_state)
    if mu_scheduler is not None and state.mu_scheduler_state is not None:
        mu_scheduler.load_state_dict(state.mu_scheduler_state)

    all_indices = list(range(n_equilibria))

    # Compute the loss at function entry. Preserve an existing initial loss when
    # continuing a run, so restart reports the original starting point.
    system.prepare_for_loss()
    with torch.no_grad():
        entry_loss_parts = _aggregate_loss_parts(
            equilibrium_states,
            all_indices,
            system,
            config=config,
            stable_weight=config.stable_weight,
            unstable_weight=config.unstable_weight,
            regularization_weight=config.regularization_weight,
            parameter0=state.parameter0,
        )
    if state.initial_loss_parts is None:
        state.initial_loss_parts = entry_loss_parts

    if not math.isfinite(state.best_loss):
        state.best_loss = float(entry_loss_parts["total"].detach().cpu())
    if checkpoint_path is not None and state.epoch == 0:
        state.update_from_runtime(
            model_optimizer,
            mu_optimizer,
            model_scheduler,
            mu_scheduler,
        )
        torch.save(system, best_model_path)
        torch.save(system, last_model_path)
        _save_equilibrium_states_without_runtime_data(
            equilibrium_states,
            best_eqstate_path,
        )
        _save_equilibrium_states_without_runtime_data(
            equilibrium_states,
            last_eqstate_path,
        )
        state.save(opt_state_path)
        console.print(
            f"saved initial checkpoint to {checkpoint_path} "
            f"(loss={state.best_loss:.2e})"
        )

    # start optimization
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
            if model_optimizer is not None:
                model_optimizer.zero_grad(set_to_none=True)
            if mu_optimizer is not None:
                mu_optimizer.zero_grad(set_to_none=True)

            system.prepare_for_loss()
            loss_parts = _aggregate_loss_parts(
                equilibrium_states,
                batch_indices,
                system,
                config=config,
                stable_weight=config.stable_weight,
                unstable_weight=config.unstable_weight,
                regularization_weight=config.regularization_weight,
                parameter0=state.parameter0,
            )
            loss_parts["total"].backward()
            if model_optimizer is not None:
                model_optimizer.step()
                if model_scheduler is not None:
                    model_scheduler.step()
            if mu_optimizer is not None:
                mu_optimizer.step()
                if mu_scheduler is not None:
                    mu_scheduler.step()

        should_record = (
            epoch == 1
            or epoch == config.epochs
            or epoch % record_every == 0
        )

        if should_record:
            system.prepare_for_loss()
            with torch.no_grad():
                epoch_loss_parts = _aggregate_loss_parts(
                    equilibrium_states,
                    all_indices,
                    system,
                    config=config,
                    stable_weight=config.stable_weight,
                    unstable_weight=config.unstable_weight,
                    regularization_weight=config.regularization_weight,
                    parameter0=state.parameter0,
                )
            state.record_history(epoch, epoch_loss_parts)
            epoch_loss = state.history["weighted_total_loss"][-1]

            stable_loss = state.history["stable_loss"][-1]
            unstable_loss = state.history["unstable_loss"][-1]
            regularization_loss = state.history["regularization_loss"][-1]
            current_lrs = {}
            if model_optimizer is not None:
                current_lrs["model"] = model_optimizer.param_groups[0]["lr"]
            if mu_optimizer is not None:
                current_lrs["latent_mu"] = mu_optimizer.param_groups[0]["lr"]
            current_lr_text = ", ".join(
                f"{name}={lr:10.2e}"
                for name, lr in current_lrs.items()
            )
            t1 = time.time()
            console.print(
                f"epoch {epoch:>4d}/{config.epochs}, "
                f"lr=({current_lr_text}), "
                f"loss={epoch_loss:10.2e}, "
                f"stable={stable_loss:10.2e}, "
                f"unstable={unstable_loss:10.2e}, "
                f"regularization={regularization_loss:10.2e}, "
                f"time={t1-t0:>.3f} sec."
            )
            t0 = t1

            if (
                config.loss_threshold is not None
                and epoch_loss <= config.loss_threshold
            ):
                console.print(
                    "\n"
                    f"stopping early at epoch {epoch}/{config.epochs}, "
                    f"loss={epoch_loss:.2e} <= "
                    f"threshold={config.loss_threshold:.2e}"
                )
                should_stop = True

            if checkpoint_path is not None and epoch_loss < state.best_loss:
                state.best_loss = epoch_loss
                state.update_from_runtime(
                    model_optimizer,
                    mu_optimizer,
                    model_scheduler,
                    mu_scheduler,
                )
                torch.save(system, best_model_path)
                _save_equilibrium_states_without_runtime_data(
                    equilibrium_states,
                    best_eqstate_path,
                )
                state.save(opt_state_path)

            if checkpoint_path is not None:
                state.update_from_runtime(
                    model_optimizer,
                    mu_optimizer,
                    model_scheduler,
                    mu_scheduler,
                )
                torch.save(system, last_model_path)
                _save_equilibrium_states_without_runtime_data(
                    equilibrium_states,
                    last_eqstate_path,
                )
                state.save(opt_state_path)

        if should_stop:
            break

    system.prepare_for_loss()
    with torch.no_grad():
        state.final_loss_parts = _aggregate_loss_parts(
            equilibrium_states,
            all_indices,
            system,
            config=config,
            stable_weight=config.stable_weight,
            unstable_weight=config.unstable_weight,
            regularization_weight=config.regularization_weight,
            parameter0=state.parameter0,
        )
    final_loss = float(state.final_loss_parts["total"].detach().cpu())
    
    if checkpoint_path is not None:
        state.update_from_runtime(
            model_optimizer,
            mu_optimizer,
            model_scheduler,
            mu_scheduler,
        )
        torch.save(system, last_model_path)
        _save_equilibrium_states_without_runtime_data(
            equilibrium_states,
            last_eqstate_path,
        )
        state.save(opt_state_path)
        console.print(f"saved latest model to {last_model_path}")
        console.print(f"saved latest equilibrium states to {last_eqstate_path}")
        console.print(f"saved latest optimization state to {opt_state_path}")
    
    if checkpoint_path is not None and final_loss < state.best_loss:
        state.best_loss = final_loss
        state.update_from_runtime(
            model_optimizer,
            mu_optimizer,
            model_scheduler,
            mu_scheduler,
        )
        torch.save(system, best_model_path)
        _save_equilibrium_states_without_runtime_data(
            equilibrium_states,
            best_eqstate_path,
        )
        state.save(opt_state_path)
        console.print(
            f"saved best model to {best_model_path} "
            f"(loss={state.best_loss:.2e})"
        )

    if state.history["weighted_total_loss"]:
        console.print(
            "initial loss: "
            f"{float(state.initial_loss_parts['total'].detach().cpu()):.2e}"
        )
        console.print(f"final   loss: {final_loss:.2e}")
        
    if print_final_results:
        console.rule("PHI AT EQUILIBRIA (FINAL)")
        _print_phi_at_equilibria(
            state.final_loss_parts,
            console=console,
        )
    console.rule("FINISHED")
    return system, equilibrium_states, state
