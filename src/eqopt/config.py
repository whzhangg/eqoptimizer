import dataclasses
import torch

@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int | None = None
    epochs: int = 1000
    lr: float = 1000.0
    latent_mu_lr: float | None = None
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam
    loss_threshold: float | None = None
    cosine_decay: bool = True
    min_lr_factor: float = 0.2

    stable_weight: float = 1.0
    unstable_weight: float = 1.0
    regularization_weight: float = 1.0e-10
    regularize_difference: bool = False

    use_huber_for_stable_phases: bool = True
    relu_margin: float = 0.0
    unstable_huber_beta: float | None = 1.0
    scale_energy_by_rt: bool = False
    composition_projection_tol: float = 1.0e-3

    mu_convergence_tol: float = 50.0
    mu_init_lr: float = 10000.0
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
