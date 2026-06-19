import math
import torch
from pathlib import Path
from typing import Sequence, Mapping
from torch import nn
import numpy as np
import dataclasses

from ...utilities import R, multi_simplex_samples_dirichlet, hit_and_run_sampling
from ...dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from ..shared import (
    get_tensor_mu,
    normalize_and_order_composition,
    scalar_temperature,
)
from ..singlephase_abc import ThermodynamicModel
from .cef_terms import (
    CEFContext,
    CEFExcessTerm,
    get_cef_context_and_terms_from_tdb_string,
)

@dataclasses.dataclass(frozen=True)
class CEFConfig:
    """Default external-facing numerical options for CEF evaluations."""

    # shared sampling/min-pooling options
    n_samples: int = 64
    use_softmin: bool = True
    softmin_tau: float | None = None
    eps: float = 1.0e-8

    # grand-potential options
    grand_potential_steps: int = 6
    grand_potential_delta: float = 0.3
    max_step_factor: float = 1.5

    # fixed-composition Gibbs-energy options
    gibbs_energy_steps: int = 6
    gibbs_energy_delta: float = 0.3

    # constrained-EGD dual solve options
    newton_steps: int = 20
    newton_damping: float = 1.0e-10
    max_dual_step: float = 2.0
    constraint_tol: float = 1.0e-8
    dual_backtracking_steps: int = 8
    primal_backtracking_steps: int = 6
    composition_penalty_weight: float | None = None


    def update(self, **kwargs) -> "CEFConfig":
        """Return a copy with selected config fields overwritten."""
        field_names = {field.name for field in dataclasses.fields(self)}
        unknown = set(kwargs) - field_names
        if unknown:
            raise TypeError(
                "Unknown CEFConfig option(s): "
                + ", ".join(sorted(unknown))
            )
        return dataclasses.replace(self, **kwargs)


class CEF(ThermodynamicModel):
    """compound energy formalism model"""
    def __init__(self,
        components_on_sublattices: Sequence[Sequence[str]],
        sublattice_multiplicities: Sequence[float],
        energy_terms: Sequence[CEFExcessTerm],
        config: CEFConfig | None = None,
        *,
        name: str | None = None,
        **kwargs,
    ):
        config = (CEFConfig() if config is None else config).update(**kwargs)

        if len(components_on_sublattices) != len(sublattice_multiplicities):
            raise ValueError(
                "components_on_sublattices and sublattice_multiplicities "
                "must have the same length."
            )
        if len(components_on_sublattices) == 0:
            raise ValueError("CEF requires at least one sublattice.")

        y_names = []
        ncomp_for_each_sublattice = []
        components = []
        for i, comps in enumerate(components_on_sublattices):
            if len(comps) == 0:
                raise ValueError(f"Sublattice {i} must contain at least one component.")
            ncomp_for_each_sublattice.append(len(comps))
            for comp in comps:
                y_names.append((comp,i))
                if comp not in components:
                    components.append(comp)

        components = tuple(components)
        elements = [c for c in components if c.upper() != "VA"]
        if len(elements) == 0:
            raise ValueError("CEF requires at least one non-vacancy element.")

        super().__init__(name or "cef", elements)
        self.config = config

        self.y_names = y_names
        y_names_to_index = {
            y_name: index
            for index, y_name in enumerate(y_names)
        }
        self.y_names_to_index = y_names_to_index
        self.ncomp_for_each_sublattice = ncomp_for_each_sublattice
        self.components = components
        self.components_on_sublattices = tuple(
            tuple(comps)
            for comps in components_on_sublattices
        )
        self.sublattice_multiplicities = tuple(float(n) for n in sublattice_multiplicities)
        self.context = CEFContext(
            y_names_to_index=y_names_to_index,
            sublattice_multiplicities=self.sublattice_multiplicities,
            phase_name=self.phase_name,
        )
        self.energy_terms = tuple(energy_terms)
        for term in self.energy_terms:
            if not isinstance(term, CEFExcessTerm):
                raise TypeError(f"Unsupported CEF energy term {type(term).__name__}.")
            term.validate(self.context)

        self.energy_interactions = nn.ModuleList(
            [term.interaction for term in self.energy_terms]
        )

        multi_for_each_y = []
        for i, nc in enumerate(self.ncomp_for_each_sublattice):
            multi_for_each_y.append(torch.ones(nc, dtype=DEFAULT_TYPE, device=DEFAULT_DEVICE)*self.sublattice_multiplicities[i])
        self.register_buffer(
            "multi_for_each_y",
            torch.cat(multi_for_each_y),
            persistent=False,
        )


    @classmethod
    def from_tdb_and_phasename(
        cls,
        tdb_path: str | Path,
        phase_name: str,
        config: CEFConfig | None = None,
        *,
        temperature_ref: float = 1000,
        correction_order: int | None = None,
        **kwargs,
    ) -> "CEF":
        phase_name = phase_name.upper()
        path = Path(tdb_path)
        if not path.exists():
            raise FileNotFoundError(f"TDB file does not exist: {path}")
        text = path.read_text()
        context, terms = get_cef_context_and_terms_from_tdb_string(
            text,
            phase_name,
            temperature_ref=temperature_ref,
            correction_order=correction_order,
        )
        components_by_sublattice: list[list[tuple[int, str]]] = [
            [] for _ in range(context.nsublattice)
        ]
        for (component, sublattice), index in context.y_names_to_index.items():
            components_by_sublattice[sublattice].append((index, component))
        components = tuple(
            tuple(component for _, component in sorted(sublattice_components))
            for sublattice_components in components_by_sublattice
        )

        return cls(
            components,
            context.sublattice_multiplicities,
            terms,
            config=config,
            name=phase_name,
            **kwargs,
        )


    def get_tdb_str(self) -> str:
        """print a TDB competible string"""
        lines = []
        nsublattice = len(self.ncomp_for_each_sublattice)
        _ncomp_str = ' '.join([str(_nc) for _nc in self.sublattice_multiplicities])
        lines.append(f'phase {self.phase_name} % {nsublattice} {_ncomp_str} !')
        comps = ' : '.join(
            [','.join([str(_c) for _c in comps]) for comps in self.components_on_sublattices])
        lines.append(f'constituent {self.phase_name} : {comps} : !')
        for term in self.energy_terms:
            lines.append(term.to_tdb_str(self.context))
        return '\n'.join(lines).upper()


    def gibbs_energy_per_molar_atom(self, comp, temperature):
        temperature = scalar_temperature(temperature)
        target_x = normalize_and_order_composition(comp, self.elements)

        if self._is_fixed_stoichiometry():
            y = self._fixed_internal_dof(target_x.shape[:-1])
            fixed_x = self._composition_from_internal_dof(y)
            if not torch.allclose(fixed_x, target_x, atol=1.0e-6, rtol=0.0):
                raise ValueError(
                    f"{self.phase_name} has fixed composition "
                    f"{fixed_x.detach().cpu().tolist()}; got "
                    f"{target_x.detach().cpu().tolist()}."
                )
            return self._energy_from_internal_dof(
                y,
                temperature,
                normalize_by_amount=True
            )

        if self._is_single_sublattice_without_vacancy():
            y = normalize_and_order_composition(comp, self.components_on_sublattices[0])
            return self._energy_from_internal_dof(
                y,
                temperature,
                normalize_by_amount=True
            )
        return self._gibbs_energy_by_constrained_EGD(comp, temperature)


    def grand_potential_per_molar_atom(self,
        mu: Mapping[str, float],
        temperature: float,
    ):
        return self._grand_potential_by_EGD(
            mu=mu,
            temperature=temperature,
            n_samples_each_side=self.config.n_samples,
            n_steps=self.config.grand_potential_steps,
            use_softmin=self.config.use_softmin,
            delta=self.config.grand_potential_delta,
            tau=self.config.softmin_tau,
            max_step_factor=self.config.max_step_factor,
            eps=self.config.eps
        )


    def sample_internal_dof_at_composition(
        self,
        composition: Mapping[str, float],
        nsamples: int,
    ) -> torch.Tensor:
        """Sample site fractions satisfying sublattice and composition constraints."""
        device = self.multi_for_each_y.device
        dtype = self.multi_for_each_y.dtype
        target_x = torch.as_tensor(
            [composition.get(element, 0.0) for element in self.elements],
            device=device,
            dtype=dtype,
        )
        target_x = target_x.clamp_min(1.0e-12)
        target_x = target_x / target_x.sum().clamp_min(1.0e-12)

        n_sublattice = len(self.sublattice_multiplicities)
        n_total_multiplicity = sum(self.sublattice_multiplicities)
        ncols = len(self.y_names)
        nrows = len(self.sublattice_multiplicities) + len(self.elements) - 1
        c_matrix = torch.zeros((nrows, ncols), device=device, dtype=dtype)
        d_matrix = torch.zeros((nrows), device=device, dtype=dtype)

        d_matrix[0:len(self.sublattice_multiplicities)] = 1.0
        ele_index = {}
        for iele, ele in enumerate(self.elements[:-1]):
            d_matrix[n_sublattice+iele] = target_x[iele] * n_total_multiplicity
            ele_index[ele] = iele

        for (comp, isub), yindex in self.context.y_names_to_index.items():
            c_matrix[isub, yindex] = 1.0
            if comp in ele_index:
                c_matrix[n_sublattice+ele_index[comp], yindex] \
                    = self.sublattice_multiplicities[isub]
            elif comp.upper() == 'VA':
                for ele in self.elements[:-1]:
                    c_matrix[n_sublattice+ele_index[ele], yindex] = (
                        self.sublattice_multiplicities[isub]
                        * target_x[ele_index[ele]]
                    )

        return hit_and_run_sampling(
            c_matrix,
            d_matrix,
            n_samples_final=nsamples,
            reduce_by_fps=True,
            n_samples_to_sample=nsamples*10
        )


    # common functions

    def _energy_from_internal_dof(self,
        y: torch.Tensor,
        temperature: float,
        normalize_by_amount: bool
    ) -> torch.Tensor:
        """return energy given input y"""
        temperature = scalar_temperature(temperature)
        y = torch.as_tensor(y, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        total = torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)

        for term in self.energy_terms:
            total += term.get_contribution(y, temperature, self.context)

        # entropy
        ylogy = R * temperature * (y * y.clamp_min(1.0e-12).log())
        total += (ylogy * self.multi_for_each_y).sum(dim=-1)

        if normalize_by_amount:
            amounts = self._get_amount_of_elements_from_y(y)
            real_atom_amount = amounts.sum(dim=-1).clamp_min(1.0e-12)
            return total / real_atom_amount
        else:
            return total


    def _composition_from_internal_dof(self, y: torch.Tensor) -> torch.Tensor:
        amounts = self._get_amount_of_elements_from_y(y)
        return amounts / amounts.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


    def _get_amount_of_elements_from_y(self, sampled_y):
        """Return unnormalized real-element amounts from CEF site fractions.

        For each real element i, M_i = sum_s N_s y_i^(s). Vacancy contributes
        zero real atoms and is therefore not included in `self.elements`.
        """
        y = torch.as_tensor(sampled_y, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        if y.shape[-1] != len(self.y_names):
            raise ValueError(
                f"Expected site fractions with {len(self.y_names)} internal "
                f"coordinates, got trailing dimension {y.shape[-1]}."
            )

        amounts = torch.zeros(
            (*y.shape[:-1], len(self.elements)),
            device=y.device,
            dtype=y.dtype,
        )
        element_to_index = {
            element: index
            for index, element in enumerate(self.elements)
        }
        for y_index, (component, sublattice_index) in enumerate(self.y_names):
            if component.upper() == "VA":
                continue
            amounts[..., element_to_index[component]] += (
                self.sublattice_multiplicities[sublattice_index] * y[..., y_index]
            )
        return amounts


    def _is_fixed_stoichiometry(self) -> bool:
        return all(n_components == 1 for n_components in self.ncomp_for_each_sublattice)


    def _grand_potential_from_internal_dof(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        amount_of_atoms = self._get_amount_of_elements_from_y(y)
        values = self._energy_from_internal_dof(
            y,
            temperature,
            normalize_by_amount=False
        ) - amount_of_atoms @ mu
        real_atom_amount = amount_of_atoms.sum(dim=-1)
        values = values / real_atom_amount.clamp_min(1.0e-12)
        return values.masked_fill(real_atom_amount <= 1.0e-12, torch.inf)


    def _fixed_internal_dof(self, batch_shape: torch.Size) -> torch.Tensor:
        return torch.ones(
            (*batch_shape, len(self.y_names)),
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )


    def _is_single_sublattice_without_vacancy(self) -> bool:
        return (
            len(self.components_on_sublattices) == 1
            and all(
                component.upper() != "VA"
                for component in self.components_on_sublattices[0]
            )
        )


    # EGD solver ############################

    def _grand_potential_by_EGD(self,
        mu: Mapping[str, float],
        temperature: float,
        n_samples_each_side: int,
        n_steps: int,
        use_softmin: bool,
        delta: float,
        tau: float | None,
        max_step_factor: int,
        eps: float
    ) -> torch.Tensor:
        """solve grand potential

        1) do a sampling according to n_samples_each_side
        2) for all points, perform exponential gradient descent for k steps
        3) from the final result, do a softmin

        The descent trajectory is detached from the outer graph. The final
        softmin recomputes grand-potential values on the refined site fractions
        so gradients still flow to thermodynamic parameters and mu.
        """
        temperature = scalar_temperature(temperature)
        mu = get_tensor_mu(mu, self.elements)

        sampled_y = multi_simplex_samples_dirichlet(
            self.ncomp_for_each_sublattice,
            n_samples_each_side,
        )
        if tau is None:
            if sampled_y.shape[0] == 1:
                tau = 1.0
            else:
                tau = 1.0 / math.log(sampled_y.shape[0]) # shape is int

        if int(n_steps) <= 0:
            values = self._grand_potential_from_internal_dof(
                sampled_y,
                mu,
                temperature,
            )
            if use_softmin:
                return -tau * torch.logsumexp(-values / tau, dim=0)
            return torch.min(values)

        y = sampled_y.detach()
        sampling_mu = mu.detach()
        eta_by_sublattice: list[torch.Tensor | None] = [
            None for _ in self.ncomp_for_each_sublattice
        ]

        with torch.enable_grad():
            total_rej = 0
            for _ in range(int(n_steps)):
                y = y.detach().requires_grad_(True)
                values = self._grand_potential_from_internal_dof(
                    y,
                    sampling_mu,
                    temperature,
                )
                grad_y = torch.autograd.grad(values.sum(), y)[0]
                grad_y = torch.nan_to_num(
                    grad_y,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                updated_parts = []
                column_start = 0
                for sublattice, n_components in enumerate(self.ncomp_for_each_sublattice):
                    column_stop = column_start + n_components
                    w = y[..., column_start:column_stop].detach()
                    g = grad_y[..., column_start:column_stop].detach()

                    if n_components == 1:
                        updated_parts.append(w)
                        column_start = column_stop
                        continue

                    g_bar = (w * g).sum(dim=-1, keepdim=True)
                    centered_g = g - g_bar
                    if eta_by_sublattice[sublattice] is None:
                        scale = (w * centered_g.abs()).amax(
                            dim=-1,
                            keepdim=True,
                        )
                        eta_by_sublattice[sublattice] = (
                            delta / scale.clamp_min(eps)
                        )

                    eta = eta_by_sublattice[sublattice]
                    proposed = self._exp_gradient_update(w, centered_g, eta)
                    actual_step = (proposed - w).abs().amax(
                        dim=-1,
                        keepdim=True,
                    )
                    too_large = actual_step > max_step_factor * delta
                    if torch.any(too_large):
                        shrink = delta / actual_step.clamp_min(eps)
                        eta = torch.where(too_large, eta * shrink, eta)
                        proposed = self._exp_gradient_update(w, centered_g, eta)
                        eta_by_sublattice[sublattice] = eta
                        total_rej += 1

                    updated_parts.append(proposed.detach())
                    column_start = column_stop

                y = torch.cat(updated_parts, dim=-1)

        values = self._grand_potential_from_internal_dof(
            y.detach(),
            mu,
            temperature,
        )
        if use_softmin:
            return -tau * torch.logsumexp(-values / tau, dim=0)
        else:
            return torch.min(values)


    def _gibbs_energy_by_constrained_EGD(self,
        composition: Mapping[str, float],
        temperature: float,
        *,
        sampled_y: torch.Tensor | None = None,
        return_internal_dof: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Estimate fixed-composition Gibbs energy with constrained EGD.

        The inner EGD trajectory is detached from the outer graph. The final
        energy is recomputed on the optimized site fractions so gradients still
        flow to thermodynamic parameters, but not through the internal solver.
        """
        temperature = scalar_temperature(temperature)
        n_samples = self.config.n_samples
        tau = self.config.softmin_tau
        use_softmin = self.config.use_softmin
        if sampled_y is None:
            sampled_initial = self.sample_internal_dof_at_composition(
                composition,
                n_samples,
            )
        else:
            sampled_initial = torch.as_tensor(
                sampled_y,
                device=self.multi_for_each_y.device,
                dtype=self.multi_for_each_y.dtype,
            )

        optimized_y = self._optimize_y_constrained_gibbs_EGD(
            sampled_initial,
            composition,
            temperature,
            n_steps=self.config.gibbs_energy_steps,
            delta=self.config.gibbs_energy_delta,
            n_newton_steps=self.config.newton_steps,
            newton_damping=self.config.newton_damping,
            max_dual_step=self.config.max_dual_step,
            constraint_tol=self.config.constraint_tol,
            max_backtracking_steps=self.config.dual_backtracking_steps,
            max_primal_backtracking_steps=self.config.primal_backtracking_steps,
            composition_penalty_weight=self.config.composition_penalty_weight,
            eps=self.config.eps
        )

        values = self._energy_from_internal_dof(
            optimized_y,
            temperature,
            normalize_by_amount=True
        )
        if use_softmin:
            if tau is None:
                if values.numel() == 1:
                    tau = 1.0
                else:
                    tau = 1.0 / math.log(values.numel())
            energy = -tau * torch.logsumexp(-values / tau, dim=0)
        else:
            energy = torch.min(values)

        if return_internal_dof:
            return energy, optimized_y.detach()
        return energy


    def _optimize_y_constrained_gibbs_EGD(
        self,
        sampled_y: torch.Tensor,
        composition: Mapping[str, float],
        temperature: float,
        *,
        n_steps: int = 6,
        delta: float = 0.3,
        n_newton_steps: int = 20,
        newton_damping: float = 1.0e-10,
        max_dual_step: float = 2.0,
        constraint_tol: float = 1.0e-8,
        max_backtracking_steps: int = 8,
        max_primal_backtracking_steps: int = 6,
        composition_penalty_weight: float | None = None,
        eps: float = 1.0e-8
    ) -> torch.Tensor:
        """Optimize feasible site fractions at fixed composition using EGD.

        Sublattice normalization is enforced by softmax-like normalization on
        each sublattice, while the supplied composition constraints are enforced
        by solving the dual variables with batched Newton iterations. A mild
        composition penalty can also be included in the search gradient to pull
        numerically drifting iterates back toward the target composition.
        """
        temperature = scalar_temperature(temperature)
        y = torch.as_tensor(
            sampled_y,
            device=self.multi_for_each_y.device,
            dtype=self.multi_for_each_y.dtype,
        )
        squeeze_output = False
        if y.ndim == 1:
            y = y.unsqueeze(0) # add a first dimension in the case when only one sample is given
            squeeze_output = True
        if y.ndim != 2 or y.shape[-1] != len(self.y_names):
            raise ValueError(
                f"Expected sampled_y with trailing dimension {len(self.y_names)}, "
                f"got shape {tuple(y.shape)}."
            )
        target_x = torch.as_tensor(
            [composition.get(element, 0.0) for element in self.elements],
            device=y.device,
            dtype=y.dtype,
        )
        target_x = target_x.clamp_min(1.0e-12)
        target_x = target_x / target_x.sum().clamp_min(1.0e-12)

        n_constraints = max(len(self.elements) - 1, 0)
        if int(n_steps) <= 0:
            return y.squeeze(0) if squeeze_output else y
        if composition_penalty_weight is None:
            composition_penalty = 10.0 * torch.abs(R * temperature).clamp_min(1.0)
        else:
            composition_penalty = torch.as_tensor(
                composition_penalty_weight,
                device=y.device,
                dtype=y.dtype,
            )

        if n_constraints:
            total_multiplicity = sum(self.sublattice_multiplicities)
            constraint_matrix = torch.zeros(
                (n_constraints, len(self.y_names)),
                device=y.device,
                dtype=y.dtype,
            )
            constraint_rhs = target_x[:n_constraints] * total_multiplicity
            element_to_constraint = {
                element: index
                for index, element in enumerate(self.elements[:-1])
            }
            for y_index, (component, sublattice_index) in enumerate(self.y_names):
                multiplicity = self.sublattice_multiplicities[sublattice_index]
                if component in element_to_constraint:
                    constraint_matrix[
                        element_to_constraint[component],
                        y_index,
                    ] = multiplicity
                elif component.upper() == "VA":
                    constraint_matrix[:, y_index] = (
                        multiplicity * target_x[:n_constraints]
                    )
        else:
            constraint_matrix = torch.empty(
                (0, len(self.y_names)),
                device=y.device,
                dtype=y.dtype,
            )
            constraint_rhs = torch.empty(0, device=y.device, dtype=y.dtype)

        optimized_y = y.detach()
        dual_mu = None
        with torch.enable_grad():
            for _ in range(int(n_steps)):
                optimized_y = optimized_y.detach().requires_grad_(True)
                values = self._energy_from_internal_dof(
                    optimized_y,
                    temperature,
                    normalize_by_amount=True
                )
                if n_constraints:
                    composition_residual = (
                        self._composition_from_internal_dof(optimized_y)
                        - target_x
                    )
                    values = values + composition_penalty * (
                        composition_residual.square().sum(dim=-1)
                    )
                grad_y = torch.autograd.grad(values.sum(), optimized_y)[0]
                grad_y = torch.nan_to_num(
                    grad_y,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                scale = torch.zeros(
                    (optimized_y.shape[0], 1),
                    device=optimized_y.device,
                    dtype=optimized_y.dtype,
                )
                column_start = 0
                for n_components in self.ncomp_for_each_sublattice:
                    column_stop = column_start + n_components
                    w = optimized_y[..., column_start:column_stop].detach()
                    g = grad_y[..., column_start:column_stop].detach()
                    g_bar = (w * g).sum(dim=-1, keepdim=True)
                    scale = torch.maximum(
                        scale,
                        (w * (g - g_bar).abs()).amax(dim=-1, keepdim=True),
                    )
                    column_start = column_stop

                eta = delta / scale.clamp_min(eps)
                previous_y = optimized_y.detach()
                previous_mu = None if dual_mu is None else dual_mu.detach()
                updated_y = previous_y
                updated_mu = previous_mu
                accepted = torch.zeros(
                    (previous_y.shape[0], 1),
                    device=previous_y.device,
                    dtype=torch.bool,
                )
                for backtrack_index in range(int(max_primal_backtracking_steps) + 1):
                    eta_try = eta * (0.5 ** backtrack_index)
                    candidate_y, candidate_mu, residual_norm = self._constrained_egd_update(
                        previous_y,
                        grad_y.detach(),
                        eta_try,
                        previous_mu,
                        self.ncomp_for_each_sublattice,
                        constraint_matrix,
                        constraint_rhs,
                        n_newton_steps=n_newton_steps,
                        newton_damping=newton_damping,
                        max_dual_step=max_dual_step,
                        constraint_tol=constraint_tol,
                        max_backtracking_steps=max_backtracking_steps,
                        eps=eps,
                    )
                    newly_accepted = (~accepted) & (residual_norm <= constraint_tol)
                    updated_y = torch.where(newly_accepted, candidate_y, updated_y)
                    if candidate_mu is not None:
                        if updated_mu is None:
                            updated_mu = torch.zeros_like(candidate_mu)
                        updated_mu = torch.where(
                            newly_accepted,
                            candidate_mu,
                            updated_mu,
                        )
                    accepted = accepted | newly_accepted
                    if bool(torch.all(accepted)):
                        break

                if bool(torch.any(accepted)):
                    optimized_y = updated_y.detach()
                    dual_mu = updated_mu
                else:
                    optimized_y = previous_y
                    dual_mu = previous_mu

        if squeeze_output:
            return optimized_y.squeeze(0)
        return optimized_y


    @staticmethod
    def _exp_gradient_update(
        w: torch.Tensor,
        centered_gradient: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        exponent = -eta * centered_gradient
        exponent = exponent - exponent.amax(dim=-1, keepdim=True)
        updated = w * torch.exp(exponent)
        updated = updated.clamp_min(1.0e-12)
        return updated / updated.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


    @staticmethod
    def _normalize_sublattice_logits(
        logits: torch.Tensor,
        ncomp_for_each_sublattice: Sequence[int],
    ) -> torch.Tensor:
        """normalize for all sublattices"""
        parts = []
        column_start = 0
        for n_components in ncomp_for_each_sublattice:
            column_stop = column_start + n_components
            parts.append(torch.softmax(logits[..., column_start:column_stop], dim=-1))
            column_start = column_stop
        return torch.cat(parts, dim=-1)


    @staticmethod
    def _constrained_egd_update(
        current_y: torch.Tensor,
        grad_y: torch.Tensor,
        eta: torch.Tensor,
        initial_mu: torch.Tensor | None,
        ncomp_for_each_sublattice: Sequence[int],
        constraint_matrix: torch.Tensor,
        constraint_rhs: torch.Tensor,
        *,
        n_newton_steps: int,
        newton_damping: float,
        max_dual_step: float,
        constraint_tol: float,
        max_backtracking_steps: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        n_constraints = constraint_matrix.shape[0]
        base_logits = current_y.clamp_min(eps).log() - eta * grad_y
        if n_constraints == 0:
            residual_norm = torch.zeros(
                (current_y.shape[0], 1),
                device=current_y.device,
                dtype=current_y.dtype,
            )
            return (
                CEF._normalize_sublattice_logits(
                    base_logits,
                    ncomp_for_each_sublattice,
                ),
                None,
                residual_norm,
            )

        batch = current_y.shape[0]
        if initial_mu is None:
            mu = torch.zeros(
                (batch, n_constraints),
                device=current_y.device,
                dtype=current_y.dtype,
            )
        else:
            mu = initial_mu.detach().clone()
        eye = torch.eye(
            n_constraints,
            device=current_y.device,
            dtype=current_y.dtype,
        )

        y_mu = current_y
        for _ in range(int(n_newton_steps)):
            y_mu, residual = CEF._y_and_constraint_residual_from_dual_mu(
                base_logits,
                mu,
                ncomp_for_each_sublattice,
                constraint_matrix,
                constraint_rhs,
            )
            residual_norm = torch.linalg.vector_norm(
                residual,
                dim=-1,
                keepdim=True,
            ) # compute residual
            if bool(torch.all(residual_norm <= constraint_tol)):
                break

            jacobian = CEF._constraint_residual_jacobian(
                y_mu,
                ncomp_for_each_sublattice,
                constraint_matrix,
            )
            linear_system = jacobian - newton_damping * eye
            rhs = -residual.unsqueeze(-1)
            try:
                delta_mu = torch.linalg.solve(linear_system, rhs).squeeze(-1)
            except RuntimeError:
                delta_mu = torch.linalg.lstsq(linear_system, rhs).solution.squeeze(-1)
            delta_mu = delta_mu.clamp(-max_dual_step, max_dual_step)

            accepted = torch.zeros(
                (batch, 1),
                device=current_y.device,
                dtype=torch.bool,
            )
            step_scale = torch.ones(
                (batch, 1),
                device=current_y.device,
                dtype=current_y.dtype,
            )
            best_mu = mu
            best_y = y_mu
            best_norm = residual_norm
            for _ in range(int(max_backtracking_steps)):
                trial_mu = mu + step_scale * delta_mu
                trial_y, trial_residual = CEF._y_and_constraint_residual_from_dual_mu(
                    base_logits,
                    trial_mu,
                    ncomp_for_each_sublattice,
                    constraint_matrix,
                    constraint_rhs,
                )
                trial_norm = torch.linalg.vector_norm(
                    trial_residual,
                    dim=-1,
                    keepdim=True,
                )
                accept = (~accepted) & (
                    (trial_norm < best_norm)
                    | (trial_norm <= constraint_tol)
                )
                best_mu = torch.where(accept, trial_mu, best_mu)
                best_y = torch.where(accept, trial_y, best_y)
                best_norm = torch.where(accept, trial_norm, best_norm)
                accepted = accepted | accept
                if bool(torch.all(accepted)):
                    break
                step_scale = torch.where(
                    accepted,
                    step_scale,
                    0.5 * step_scale,
                )

            mu = best_mu
            y_mu = best_y

        final_residual = y_mu @ constraint_matrix.mT - constraint_rhs
        final_residual_norm = torch.linalg.vector_norm(
            final_residual,
            dim=-1,
            keepdim=True,
        )
        return y_mu, mu, final_residual_norm


    @staticmethod
    def _y_and_constraint_residual_from_dual_mu(
        base_logits: torch.Tensor,
        dual_mu: torch.Tensor,
        ncomp_for_each_sublattice: Sequence[int],
        constraint_matrix: torch.Tensor,
        constraint_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = base_logits - dual_mu @ constraint_matrix
        candidate_y = CEF._normalize_sublattice_logits(
            logits,
            ncomp_for_each_sublattice,
        )
        candidate_residual = candidate_y @ constraint_matrix.mT - constraint_rhs
        return candidate_y, candidate_residual


    @staticmethod
    def _constraint_residual_jacobian(
        y: torch.Tensor,
        ncomp_for_each_sublattice: Sequence[int],
        constraint_matrix: torch.Tensor,
    ) -> torch.Tensor:
        batch = y.shape[0]
        n_constraints = constraint_matrix.shape[0]
        jacobian = torch.zeros(
            (batch, n_constraints, n_constraints),
            device=y.device,
            dtype=y.dtype,
        )
        column_start = 0
        for n_components in ncomp_for_each_sublattice:
            column_stop = column_start + n_components
            w = y[..., column_start:column_stop]
            local_a = constraint_matrix[:, column_start:column_stop]
            mean_a = w @ local_a.mT
            second_moment = torch.einsum(
                "bi,ai,ci->bac",
                w,
                local_a,
                local_a,
            )
            jacobian = jacobian + (
                mean_a[:, :, None] * mean_a[:, None, :]
                - second_moment
            )
            column_start = column_stop
        return jacobian


    # Scipy solver #########################

    def grand_potential_per_molar_atom_by_scipy(self,
        mu: Mapping[str, float],
        temperature: float,
        *,
        n_samples_to_optimize: int = 8,
        max_iter: int = 300,
        tol: float = 1.0e-5
    ) -> torch.Tensor:
        """solve the grand potential using scipy."""
        from scipy.optimize import minimize

        temperature = scalar_temperature(temperature)
        mu = get_tensor_mu(mu, self.elements)
        reduced_dim = sum(n - 1 for n in self.ncomp_for_each_sublattice)

        if reduced_dim == 0:
            y = self._fixed_internal_dof(torch.Size())
            return self._grand_potential_from_internal_dof(y, mu, temperature)

        def tensor_from_numpy(
            logits_np: np.ndarray,
            *,
            requires_grad: bool,
        ) -> torch.Tensor:
            logits = torch.as_tensor(
                logits_np,
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            logits = logits.detach()
            if requires_grad:
                logits = logits.requires_grad_(True)
            return logits

        def objective(logits_np: np.ndarray) -> tuple[float, np.ndarray]:
            with torch.enable_grad():
                logits = tensor_from_numpy(logits_np, requires_grad=True)
                y = self._internal_dof_from_reduced_logits(logits)
                value = self._grand_potential_from_internal_dof(
                    y,
                    mu.detach(),
                    temperature,
                ).reshape(())
                grad = torch.autograd.grad(value, logits)[0]
            return (
                float(value.detach().cpu()),
                grad.detach().cpu().numpy(),
            )

        sampled_y = multi_simplex_samples_dirichlet(
            self.ncomp_for_each_sublattice,
            self.config.n_samples,
        )
        with torch.no_grad():
            sampled_values = self._grand_potential_from_internal_dof(
                sampled_y,
                mu.detach(),
                temperature,
            )
            n_starts = min(n_samples_to_optimize, sampled_y.shape[0])
            start_indices = torch.topk(
                sampled_values,
                k=n_starts,
                largest=False,
            ).indices
            starts = self._reduced_logits_from_internal_dof(
                sampled_y[start_indices]
            )
            starts = torch.cat([
                torch.zeros(
                    (1, reduced_dim),
                    device=DEFAULT_DEVICE,
                    dtype=DEFAULT_TYPE,
                ),
                starts,
            ], dim=0)

        best_result = None
        best_value = float("inf")
        for start in starts:
            result = minimize(
                objective,
                start.detach().cpu().numpy(),
                method="BFGS",
                jac=True,
                options={
                    "gtol": tol,
                    "maxiter": max_iter,
                    "disp": False,
                },
            )
            value = float(result.fun)
            if value < best_value:
                best_value = value
                best_result = result

        if best_result is None:
            raise RuntimeError(
                f"SciPy grand-potential minimization failed for {self.phase_name}: "
                "no starts were generated."
            )

        logits = tensor_from_numpy(best_result.x, requires_grad=False)
        y = self._internal_dof_from_reduced_logits(logits)
        return self._grand_potential_from_internal_dof(y, mu, temperature)


    def gibbs_energy_per_molar_atom_by_scipy(self,
        comp: Mapping[str, float],
        temperature: float,
        *,
        max_iter: int = 300,
        tol: float = 1.0e-5
    ):
        """constrained minimization"""
        from scipy.optimize import minimize
        temperature = scalar_temperature(temperature)
        target_x = normalize_and_order_composition(comp, self.elements)

        if target_x.ndim < 1 or target_x.shape[-1] != len(self.elements):
            raise ValueError(
                "SciPy constrained CEF minimization expects compositions with "
                f"trailing dimension {len(self.elements)}; got shape "
                f"{tuple(target_x.shape)}."
            )

        batch_shape = target_x.shape[:-1]
        flat_target_x = target_x.reshape(-1, len(self.elements))
        n_constraints = max(len(self.elements) - 1, 0)

        def tensor_from_numpy(
            logits_np: np.ndarray,
            *,
            requires_grad: bool,
        ) -> torch.Tensor:
            logits = torch.as_tensor(
                logits_np,
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            )
            logits = logits.detach()
            if requires_grad:
                logits = logits.requires_grad_(True)
            return logits

        def solve_one(single_target_x: torch.Tensor) -> torch.Tensor:
            single_target_x = single_target_x.detach()
            initial_logits = self._reduced_initial_logits_from_composition(
                single_target_x
            )
            x0 = initial_logits.detach().cpu().numpy()
            objective_scale = torch.clamp(
                torch.abs(R * temperature),
                min=1.0,
            )

            def objective(logits_np: np.ndarray) -> tuple[float, np.ndarray]:
                with torch.enable_grad():
                    logits = tensor_from_numpy(logits_np, requires_grad=True)
                    y = self._internal_dof_from_reduced_logits(logits)
                    value = self._energy_from_internal_dof(
                        y,
                        temperature,
                        normalize_by_amount=True
                    ).reshape(()) / objective_scale
                    grad = torch.autograd.grad(value, logits)[0]
                return (
                    float(value.detach().cpu()),
                    grad.detach().cpu().numpy(),
                )

            def constraint_value(logits_np: np.ndarray) -> np.ndarray:
                if n_constraints == 0:
                    return np.empty(0, dtype=float)
                logits = tensor_from_numpy(logits_np, requires_grad=False)
                y = self._internal_dof_from_reduced_logits(logits)
                x = self._composition_from_internal_dof(y)
                residual = (
                    x[:n_constraints]
                    - single_target_x[:n_constraints]
                )
                return residual.detach().cpu().numpy()

            def constraint_jacobian(logits_np: np.ndarray) -> np.ndarray:
                if n_constraints == 0:
                    return np.empty((0, len(logits_np)), dtype=float)
                with torch.enable_grad():
                    logits = tensor_from_numpy(logits_np, requires_grad=True)
                    y = self._internal_dof_from_reduced_logits(logits)
                    x = self._composition_from_internal_dof(y)
                    residual = (
                        x[:n_constraints]
                        - single_target_x[:n_constraints]
                    )
                    rows = []
                    for index in range(n_constraints):
                        grad = torch.autograd.grad(
                            residual[index],
                            logits,
                            retain_graph=index < n_constraints - 1,
                        )[0]
                        rows.append(grad.detach().cpu().numpy())
                return np.stack(rows, axis=0)

            constraints = []
            if n_constraints:
                constraints.append({
                    "type": "eq",
                    "fun": constraint_value,
                    "jac": constraint_jacobian,
                })

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                jac=True,
                constraints=constraints,
                options={
                    "ftol": tol,
                    "maxiter": max_iter,
                    "disp": False,
                },
            )
            logits = tensor_from_numpy(result.x, requires_grad=False)
            y = self._internal_dof_from_reduced_logits(logits)
            x = self._composition_from_internal_dof(y)
            residual = torch.max(torch.abs(x - single_target_x)).item()
            if not result.success and residual > 10.0 * tol:
                raise RuntimeError(
                    f"SciPy constrained CEF minimization failed for "
                    f"{self.phase_name}: {result.message}; max composition "
                    f"residual={residual:.3e}."
                )
            return self._energy_from_internal_dof(
                y,
                temperature,
                normalize_by_amount=True
            )

        values = [solve_one(single_target_x) for single_target_x in flat_target_x]
        return torch.stack(values, dim=0).reshape(batch_shape)


    def _internal_dof_from_reduced_logits(
        self,
        reduced_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return site fractions from logits with one reference per sublattice."""
        reduced_logits = torch.as_tensor(
            reduced_logits,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        parts = []
        column_start = 0
        for n_components in self.ncomp_for_each_sublattice:
            if n_components == 1:
                parts.append(
                    torch.ones(
                        (*reduced_logits.shape[:-1], 1),
                        device=reduced_logits.device,
                        dtype=reduced_logits.dtype,
                    )
                )
                continue
            column_stop = column_start + n_components - 1
            sublattice_logits = reduced_logits[..., column_start:column_stop]
            reference = torch.zeros(
                (*reduced_logits.shape[:-1], 1),
                device=reduced_logits.device,
                dtype=reduced_logits.dtype,
            )
            parts.append(
                torch.softmax(
                    torch.cat([sublattice_logits, reference], dim=-1),
                    dim=-1,
                )
            )
            column_start = column_stop
        return torch.cat(parts, dim=-1)


    def _reduced_logits_from_internal_dof(self, y: torch.Tensor) -> torch.Tensor:
        """Return reduced logits whose reference component is last per sublattice."""
        y = torch.as_tensor(y, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        parts = []
        column_start = 0
        for n_components in self.ncomp_for_each_sublattice:
            column_stop = column_start + n_components
            if n_components > 1:
                sublattice_y = y[..., column_start:column_stop].clamp_min(1.0e-12)
                parts.append(
                    sublattice_y[..., :-1].log()
                    - sublattice_y[..., -1:].log()
                )
            column_start = column_stop
        if not parts:
            return torch.empty(
                (*y.shape[:-1], 0),
                device=y.device,
                dtype=y.dtype,
            )
        return torch.cat(parts, dim=-1)


    def _initial_logits_from_composition(self, target_x: torch.Tensor) -> torch.Tensor:
        target_x = torch.as_tensor(target_x, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        element_to_index = {
            element: index
            for index, element in enumerate(self.elements)
        }
        initial_parts = []
        for sublattice_components in self.components_on_sublattices:
            weights = []
            for component in sublattice_components:
                if component.upper() == "VA":
                    weights.append(
                        torch.full(
                            target_x.shape[:-1],
                            1.0e-6,
                            device=target_x.device,
                            dtype=target_x.dtype,
                        )
                    )
                elif component in element_to_index:
                    weights.append(target_x[..., element_to_index[component]])
                else:
                    weights.append(
                        torch.zeros(
                            target_x.shape[:-1],
                            device=target_x.device,
                            dtype=target_x.dtype,
                        )
                    )

            sublattice_y = torch.stack(weights, dim=-1)
            total = sublattice_y.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(
                sublattice_y,
                1.0 / len(sublattice_components),
            )
            sublattice_y = torch.where(
                total > 1.0e-12,
                sublattice_y / total.clamp_min(1.0e-12),
                uniform,
            )
            initial_parts.append(sublattice_y)
        return torch.cat(initial_parts, dim=-1).clamp_min(1.0e-12).log()


    def _reduced_initial_logits_from_composition(
        self,
        target_x: torch.Tensor,
    ) -> torch.Tensor:
        full_logits = self._initial_logits_from_composition(target_x)
        return self._reduced_logits_from_internal_dof(
            self._internal_dof_from_logits(full_logits)
        )


    def _internal_dof_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.as_tensor(logits, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        parts = []
        column_start = 0
        for n_components in self.ncomp_for_each_sublattice:
            column_stop = column_start + n_components
            parts.append(torch.softmax(logits[..., column_start:column_stop], dim=-1))
            column_start = column_stop
        return torch.cat(parts, dim=-1)
