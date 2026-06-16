import math
import torch
from pathlib import Path
from typing import Sequence, Mapping
from torch import nn
import numpy as np

from ...utilities import R, multi_simplex_samples_dirichlet
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


class CEF(ThermodynamicModel):
    cef_max_iter = 300
    cef_tol = 1.0e-5
    cef_eps = 1.0e-8
    def __init__(self, 
        components_on_sublattices: Sequence[Sequence[str]],
        sublattice_multiplicities: Sequence[float],
        energy_terms: Sequence[CEFExcessTerm],
        *,
        name: str | None = None,
    ):
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
        self._validate_energy_terms()
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


    def _validate_energy_terms(self) -> None:
        for term in self.energy_terms:
            if not isinstance(term, CEFExcessTerm):
                raise TypeError(f"Unsupported CEF energy term {type(term).__name__}.")
            term.validate(self.context)


    def _internal_dof_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.as_tensor(logits, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        parts = []
        column_start = 0
        for n_components in self.ncomp_for_each_sublattice:
            column_stop = column_start + n_components
            parts.append(torch.softmax(logits[..., column_start:column_stop], dim=-1))
            column_start = column_stop
        return torch.cat(parts, dim=-1)


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


    def _composition_from_internal_dof(self, y: torch.Tensor) -> torch.Tensor:
        amounts = self.get_amount_of_elements_from_y(y)
        return amounts / amounts.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


    def _gibbs_energy_per_molar_atom_from_internal_dof(
        self,
        y: torch.Tensor,
        temperature,
    ) -> torch.Tensor:
        amounts = self.get_amount_of_elements_from_y(y)
        real_atom_amount = amounts.sum(dim=-1).clamp_min(1.0e-12)
        return self._gibbs_energy_from_internal_dof(y, temperature) / real_atom_amount


    def _fixed_internal_dof(self, batch_shape: torch.Size) -> torch.Tensor:
        return torch.ones(
            (*batch_shape, len(self.y_names)),
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )


    def _is_fixed_stoichiometry(self) -> bool:
        return all(n_components == 1 for n_components in self.ncomp_for_each_sublattice)


    def _is_single_sublattice_without_vacancy(self) -> bool:
        return (
            len(self.components_on_sublattices) == 1
            and all(
                component.upper() != "VA"
                for component in self.components_on_sublattices[0]
            )
        )


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
    

    def _gibbs_energy_from_internal_dof(self, y: torch.Tensor, temperature: float):
        temperature = scalar_temperature(temperature)
        y = torch.as_tensor(y, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        total = torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)

        for term in self.energy_terms:
            total += term.get_contribution(y, temperature, self.context)

        # entropy
        ylogy = R * temperature * (y * y.clamp_min(1.0e-12).log())
        total += (ylogy * self.multi_for_each_y).sum(dim=-1)

        return total
    

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
            return self._gibbs_energy_per_molar_atom_from_internal_dof(
                y,
                temperature,
            )

        if self._is_single_sublattice_without_vacancy():
            y = normalize_and_order_composition(comp, self.components_on_sublattices[0])
            return self._gibbs_energy_per_molar_atom_from_internal_dof(
                y,
                temperature,
            )

        return self._solve_constrained_gibbs_scipy(target_x, temperature)
    

    def _solve_constrained_gibbs_scipy(self, target_x, temperature: torch.Tensor):
        """constrained minimization"""
        from scipy.optimize import minimize

        target_x = torch.as_tensor(target_x, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
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
                    value = self._gibbs_energy_per_molar_atom_from_internal_dof(
                        y,
                        temperature,
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
                    "ftol": self.cef_tol,
                    "maxiter": self.cef_max_iter,
                    "disp": False,
                },
            )
            logits = tensor_from_numpy(result.x, requires_grad=False)
            y = self._internal_dof_from_reduced_logits(logits)
            x = self._composition_from_internal_dof(y)
            residual = torch.max(torch.abs(x - single_target_x)).item()
            if not result.success and residual > 10.0 * self.cef_tol:
                raise RuntimeError(
                    f"SciPy constrained CEF minimization failed for "
                    f"{self.phase_name}: {result.message}; max composition "
                    f"residual={residual:.3e}."
                )
            return self._gibbs_energy_per_molar_atom_from_internal_dof(
                y,
                temperature,
            )

        values = [solve_one(single_target_x) for single_target_x in flat_target_x]
        return torch.stack(values, dim=0).reshape(batch_shape)


    def get_amount_of_elements_from_y(self, sampled_y):
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

    
    def grand_potential_per_molar_atom(self, 
        mu: Mapping[str, float], 
        temperature: float, 
        tau: float | None = None, 
        *,
        n_samples_each_side = 64,
        n_steps: int = 6,
        delta: float = 0.3,
        max_step_factor: float = 1.5
    ):
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
            return -tau * torch.logsumexp(-values / tau, dim=0)

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
                            delta / scale.clamp_min(self.cef_eps)
                        )

                    eta = eta_by_sublattice[sublattice]
                    proposed = self._exp_gradient_update(w, centered_g, eta)
                    actual_step = (proposed - w).abs().amax(
                        dim=-1,
                        keepdim=True,
                    )
                    too_large = actual_step > max_step_factor * delta
                    if torch.any(too_large):
                        shrink = delta / actual_step.clamp_min(self.cef_eps)
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
        return -tau * torch.logsumexp(-values / tau, dim=0)


    def _exp_gradient_update(
        self,
        w: torch.Tensor,
        centered_gradient: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        exponent = -eta * centered_gradient
        exponent = exponent - exponent.amax(dim=-1, keepdim=True)
        updated = w * torch.exp(exponent)
        updated = updated.clamp_min(1.0e-12)
        return updated / updated.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


    def _grand_potential_from_internal_dof(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        amount_of_atoms = self.get_amount_of_elements_from_y(y)
        values = self._gibbs_energy_from_internal_dof(
            y,
            temperature,
        ) - amount_of_atoms @ mu
        real_atom_amount = amount_of_atoms.sum(dim=-1)
        values = values / real_atom_amount.clamp_min(1.0e-12)
        return values.masked_fill(real_atom_amount <= 1.0e-12, torch.inf)


    def grand_potential_per_molar_atom_by_scipy(self,
        mu: Mapping[str, float], 
        temperature: float, 
        *,
        n_samples_each_side = 64
    ):
        """solve the grand potential using scipy."""
        temperature = scalar_temperature(temperature)
        mu = get_tensor_mu(mu, self.elements)
        return self._solve_grand_potential_by_scipy(
            mu, temperature, n_samples_each_side=n_samples_each_side
        )
    

    def _solve_grand_potential_by_scipy(
        self,
        mu: torch.Tensor,
        temperature: torch.Tensor,
        *,
        n_samples_each_side: int = 64,
        n_samples_to_optimize: int = 8
    ) -> torch.Tensor:
        """
        Minimize grand potential over internal degrees of freedom.
        
        1) make a overall sampling
        2) select top n_samples to do subsequent minimization BFGS
        3) return the best result with minimal gibbs energy
        """
        from scipy.optimize import minimize

        mu = torch.as_tensor(mu, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
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
            n_samples_each_side,
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
                    "gtol": self.cef_tol,
                    "maxiter": self.cef_max_iter,
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


    def get_tdb_str(self):
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
    
    
    @classmethod
    def from_tdb_and_phasename(
        cls, 
        tdb_path: str | Path, 
        phase_name: str, 
        *,
        temperature_ref: float = 1000,
        correction_order: int | None = None,
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
            name=phase_name,
        )
