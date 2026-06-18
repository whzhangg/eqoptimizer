import dataclasses
import torch

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
    use_softmin: bool = True
    relu_margin: float = 0.0
    unstable_huber_beta: float | None = 1.0
    scale_energy_by_rt: bool = True
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
