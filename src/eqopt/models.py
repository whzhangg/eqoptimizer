from abc import ABC, abstractmethod
from typing import Sequence
import sys

import torch
from torch import nn, Tensor

from .dtype import TORCH_FLOAT
from .utilities import as_float_tensor, R


class GibbsModel(nn.Module, ABC):
    """Base class for differentiable thermodynamic phase models.

    Subclasses define Gibbs energy at an imposed composition. The base class
    stores phase metadata and provides the shared soft-min grand potential.
    """

    def __init__(self, name: str, elements: Sequence[str]) -> None:
        super().__init__()
        if not elements:
            raise ValueError("A GibbsModel requires at least one element.")
        self.name = name
        self.elements = tuple(elements)


    @property
    def n_components(self) -> int:
        return len(self.elements)


    @property
    def phase_name(self) -> str:
        return self.name


    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")


    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            try:
                return next(self.buffers()).dtype
            except StopIteration:
                return TORCH_FLOAT


    def normalize_composition(self, x) -> Tensor:
        x = as_float_tensor(x, device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.n_components:
            raise ValueError(
                f"Expected composition with {self.n_components} components "
                f"({self.elements}), got {x.shape[-1]}."
            )
        x = x.clamp_min(1.0e-12)
        return x / x.sum(dim=-1, keepdim=True)


    @abstractmethod
    def gibbs_energy(self, x, temperature) -> Tensor:
        """Return molar Gibbs energy at imposed composition and temperature."""


    def forward(self, x, temperature) -> Tensor:
        return self.gibbs_energy(x, temperature)


    def grand_potential(self, mu, temperature, samples: Tensor, tau: float = 1.0) -> Tensor:
        """Soft-min of G(x, T) - mu dot x over supplied composition samples."""
        mu = as_float_tensor(mu, device=self.device, dtype=self.dtype)
        samples = self.normalize_composition(samples)
        if mu.shape[-1] != self.n_components:
            raise ValueError(
                f"Expected chemical potential with {self.n_components} components "
                f"({self.elements}), got {mu.shape[-1]}."
            )
        values = self.gibbs_energy(samples, temperature) - samples @ mu
        return -tau * torch.logsumexp(-values / tau, dim=0)


class SolidSolutionModel(GibbsModel):
    """Substitutional solid solution Gibbs-energy model.

    G(x, T) = sum_i x_i G_i(T)
            #+ R T sum_i x_i log(x_i)
            + sum_{i<j} x_i x_j sum_n
              (a_ij^(n) + b_ij^(n) T / T_ref) (x_i - x_j)^n
    """

    def __init__(
        self,
        n_components: int,
        polynomial_order: int = 4,
        interaction_order: int = 0,
        *,
        name: str | None = None,
        elements: Sequence[str] | None = None,
        init_scale: float = 1.0e3,
        temperature_ref: float = 1000.0,
        device=None,
    ) -> None:
        if elements is None:
            elements = tuple(f"C{i}" for i in range(n_components))
        if len(elements) != n_components:
            raise ValueError(
                f"Expected {n_components} elements, got {len(elements)}."
            )
        super().__init__(name or "solid_solution", elements)
        if n_components < 2:
            raise ValueError("SolidSolutionModel requires at least two components.")

        self.polynomial_order = polynomial_order
        self.interaction_order = interaction_order
        self.temperature_ref = float(temperature_ref)

        kwargs = {"device": device, "dtype": TORCH_FLOAT}
        self.endmember_coeffs = nn.Parameter(
            init_scale
            * torch.randn(n_components, polynomial_order + 1, **kwargs)
            / (polynomial_order + 1) ** 0.5
        )

        pair_indices = torch.combinations(torch.arange(n_components), r=2)
        self.register_buffer("pair_indices", pair_indices, persistent=False)
        self.interaction_coeffs = nn.Parameter(
            init_scale
            * torch.randn(pair_indices.shape[0], interaction_order + 1, 2, **kwargs)
            / (interaction_order + 1) ** 0.5
        )


    def pure_component_gibbs(self, temperature) -> Tensor:
        temperature = as_float_tensor(
            temperature, device=self.device, dtype=self.dtype
        )
        powers = torch.arange(
            self.polynomial_order + 1, device=self.device, dtype=self.dtype
        )
        t_powers = (temperature[..., None]/self.temperature_ref) ** powers
        return t_powers @ self.endmember_coeffs.T


    def gibbs_energy(self, x, temperature) -> Tensor:
        x = self.normalize_composition(x)
        temperature = as_float_tensor(
            temperature, device=self.device, dtype=self.dtype
        )

        pure = (x * self.pure_component_gibbs(temperature)).sum(dim=-1)
        #ideal = R * temperature * (x * x.log()).sum(dim=-1)

        xi = x[..., self.pair_indices[:, 0]]
        xj = x[..., self.pair_indices[:, 1]]
        delta = xi - xj
        order = torch.arange(
            self.interaction_order + 1, device=self.device, dtype=self.dtype
        )
        interaction = (
            self.interaction_coeffs[..., 0]
            + self.interaction_coeffs[..., 1]
            * (temperature / self.temperature_ref
            )[..., None, None]
        )
        rk_terms = (delta[..., None] ** order) * interaction
        excess = (xi * xj * rk_terms.sum(dim=-1)).sum(dim=-1)

        return pure + excess

    def parameter_report(self) -> str:
        """Return a readable summary of optimized thermodynamic parameters."""
        lines = [
            f"SolidSolutionModel(name={self.phase_name!r}, elements={self.elements})",
            f"temperature_ref = {self.temperature_ref:g}",
            "endmember_coeffs:",
        ]
        endmember = self.endmember_coeffs.detach().cpu()
        for element, coeffs in zip(self.elements, endmember):
            values = ", ".join(
                f"c{order}={float(value):.8g}"
                for order, value in enumerate(coeffs)
            )
            lines.append(f"  {element}: {values}")

        lines.append("interaction_coeffs:")
        interactions = self.interaction_coeffs.detach().cpu()
        pair_indices = self.pair_indices.detach().cpu()
        for pair_index, (i, j) in enumerate(pair_indices.tolist()):
            pair = f"{self.elements[i]}-{self.elements[j]}"
            lines.append(f"  {pair}:")
            for order, coeffs in enumerate(interactions[pair_index]):
                a, b = coeffs.tolist()
                lines.append(f"    n={order}: a={a:.8g}, b={b:.8g}")
        return "\n".join(lines)

    def print_parameters(self, file=None) -> None:
        """Print a readable summary of optimized thermodynamic parameters."""
        print(self.parameter_report(), file=file or sys.stdout)


class PycalphadGibbsModel(GibbsModel):
    """Fixed GibbsModel wrapper around a PycalphadReferenceModel.

    This model exposes pycalphad reference energies through the same interface
    as trainable torch models, but it has no trainable parameters.
    """

    def __init__(
        self,
        reference_model,
        *,
        name: str | None = None,
        device=None,
    ) -> None:
        super().__init__(
            name or getattr(reference_model, "phase_name", "pycalphad_reference"),
            reference_model.elements,
        )
        self.reference_model = reference_model
        reference_device = device or getattr(reference_model, "device", None)
        self.register_buffer(
            "_reference_tensor",
            torch.empty((), dtype=TORCH_FLOAT, device=reference_device),
            persistent=False,
        )

    def _reference_temperature(self, temperature) -> float:
        temperature_tensor = as_float_tensor(
            temperature,
            device=self.device,
            dtype=self.dtype,
        ).detach().cpu().reshape(-1)
        if temperature_tensor.numel() != 1:
            raise ValueError(
                "PycalphadGibbsModel reference evaluation currently requires "
                "a scalar temperature."
            )
        return float(temperature_tensor[0])

    def gibbs_energy(self, x, temperature) -> Tensor:
        input_x = as_float_tensor(x, device=self.device, dtype=self.dtype)
        single_composition = input_x.ndim == 1
        x = self.normalize_composition(input_x)
        reference = self.reference_model.gibbs_energy(
            x.detach().cpu().numpy(),
            self._reference_temperature(temperature),
        )
        gibbs = as_float_tensor(
            reference.gibbs_energy,
            device=x.device,
            dtype=x.dtype,
        )
        if single_composition:
            return gibbs.reshape(())
        return gibbs

    def grand_potential(
        self,
        mu,
        temperature,
        samples=None,
        tau: float = 1.0,
        n_samples_each_side: int = 16,
    ) -> Tensor:
        """Soft-min of fixed pycalphad G0(y) - mu dot X(y)."""
        reference_temperature = self._reference_temperature(temperature)
        if samples is None:
            samples = self.reference_model.sampled_internal_dof(
                reference_temperature,
                n_samples_each_side=n_samples_each_side,
            )

        if hasattr(samples, "gibbs_energy") and hasattr(samples, "x"):
            x = as_float_tensor(samples.x, device=self.device, dtype=self.dtype)
            reference_gibbs = as_float_tensor(
                samples.gibbs_energy,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            x = self.normalize_composition(samples)
            reference = self.reference_model.gibbs_energy(
                x.detach().cpu().numpy(),
                reference_temperature,
            )
            reference_gibbs = as_float_tensor(
                reference.gibbs_energy,
                device=x.device,
                dtype=x.dtype,
            )

        mu = as_float_tensor(mu, device=x.device, dtype=x.dtype)
        if mu.shape[-1] != self.n_components:
            raise ValueError(
                f"Expected chemical potential with {self.n_components} components "
                f"({self.elements}), got {mu.shape[-1]}."
            )
        values = reference_gibbs - x @ mu
        return -tau * torch.logsumexp(-values / tau, dim=0)


class CorrectedGibbsModel(GibbsModel):
    """Pycalphad reference model plus a trainable Gibbs-energy correction.

    The reference model is treated as fixed and non-differentiable. Gradients
    flow through `correction_model` only.
    """

    def __init__(
        self,
        reference_model,
        correction_model: GibbsModel,
        *,
        name: str | None = None,
    ) -> None:
        if tuple(reference_model.elements) != tuple(correction_model.elements):
            raise ValueError(
                "Reference and correction models must use the same element order. "
                f"Got {reference_model.elements} and {correction_model.elements}."
            )
        super().__init__(
            name or getattr(reference_model, "phase_name", correction_model.phase_name),
            correction_model.elements,
        )
        self.reference_model = reference_model
        self.correction_model = correction_model

    def _reference_temperature(self, temperature) -> float:
        temperature_tensor = as_float_tensor(temperature).detach().cpu().reshape(-1)
        if temperature_tensor.numel() != 1:
            raise ValueError(
                "CorrectedGibbsModel reference evaluation currently requires "
                "a scalar temperature."
            )
        return float(temperature_tensor[0])

    def gibbs_energy(self, x, temperature) -> Tensor:
        input_x = as_float_tensor(x, device=self.device, dtype=self.dtype)
        single_composition = input_x.ndim == 1
        x = self.normalize_composition(input_x)
        reference_temperature = self._reference_temperature(temperature)
        reference = self.reference_model.gibbs_energy(
            x.detach().cpu().numpy(),
            reference_temperature,
        )
        reference_gibbs = as_float_tensor(
            reference.gibbs_energy,
            device=x.device,
            dtype=x.dtype,
        )
        gibbs = reference_gibbs + self.correction_model(x, temperature)
        if single_composition:
            return gibbs.reshape(())
        return gibbs

    def grand_potential(
        self,
        mu,
        temperature,
        samples=None,
        tau: float = 1.0,
        n_samples_each_side: int = 16,
    ) -> Tensor:
        """Soft-min of G0(y) + Gcorr(X(y)) - mu dot X(y)."""
        reference_temperature = self._reference_temperature(temperature)
        if samples is None:
            samples = self.reference_model.sampled_internal_dof(
                reference_temperature,
                n_samples_each_side=n_samples_each_side,
            )

        if hasattr(samples, "gibbs_energy") and hasattr(samples, "x"):
            x = as_float_tensor(
                samples.x,
                device=self.device,
                dtype=self.dtype,
            )
            reference_gibbs = as_float_tensor(
                samples.gibbs_energy,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            x = self.normalize_composition(samples)
            reference = self.reference_model.gibbs_energy(
                x.detach().cpu().numpy(),
                reference_temperature,
            )
            reference_gibbs = as_float_tensor(
                reference.gibbs_energy,
                device=x.device,
                dtype=x.dtype,
            )

        mu = as_float_tensor(mu, device=x.device, dtype=x.dtype)
        if mu.shape[-1] != self.n_components:
            raise ValueError(
                f"Expected chemical potential with {self.n_components} components "
                f"({self.elements}), got {mu.shape[-1]}."
            )
        values = (
            reference_gibbs
            + self.correction_model(x, temperature)
            - x @ mu
        )
        return -tau * torch.logsumexp(-values / tau, dim=0)
