from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .dtype import TORCH_FLOAT
from .models import GibbsModel
from .utilities import R, as_float_tensor


@dataclasses.dataclass(frozen=True)
class SampledSurface:
    """Sampled composition-energy points from a GibbsModel."""

    x: Tensor
    temperature: Tensor
    gibbs_energy: Tensor


def _as_temperature_sequence(temperatures) -> list[float]:
    if np.isscalar(temperatures):
        return [float(temperatures)]
    return [float(temperature) for temperature in temperatures]


def _samples_to_composition(samples, model: GibbsModel) -> Tensor:
    if hasattr(samples, "x"):
        return as_float_tensor(samples.x, device=model.device, dtype=model.dtype)
    return model.normalize_composition(samples)


def _sampled_gibbs_energy(model: GibbsModel, samples, x: Tensor, temperature: float) -> Tensor:
    if hasattr(samples, "gibbs_energy"):
        reference_gibbs = as_float_tensor(
            samples.gibbs_energy,
            device=model.device,
            dtype=model.dtype,
        ).reshape(-1)
        correction_model = getattr(model, "correction_model", None)
        if correction_model is not None:
            return reference_gibbs + correction_model(x, temperature).reshape(-1)
        return reference_gibbs
    return model.gibbs_energy(x, temperature).reshape(-1)


def sample_model_surface(
    model: GibbsModel,
    temperatures,
    *,
    n_samples_each_side: int = 16,
) -> SampledSurface:
    """Sample a model and evaluate G at the sampled compositions.

    For pycalphad-backed models this samples their internal degrees of freedom
    and uses the resulting compositions. For ordinary composition models this
    samples the composition simplex.
    """
    xs = []
    temperature_values = []
    gibbs_values = []

    with torch.no_grad():
        for temperature in _as_temperature_sequence(temperatures):
            samples = model.sample_degree_of_freedom(
                temperature,
                n_samples_each_side=n_samples_each_side,
            )
            x = _samples_to_composition(samples, model)
            g = _sampled_gibbs_energy(model, samples, x, temperature)
            xs.append(x.detach())
            gibbs_values.append(g.detach())
            temperature_values.append(
                torch.full(
                    (x.shape[0],),
                    float(temperature),
                    device=x.device,
                    dtype=x.dtype,
                )
            )

    return SampledSurface(
        x=torch.cat(xs, dim=0),
        temperature=torch.cat(temperature_values, dim=0),
        gibbs_energy=torch.cat(gibbs_values, dim=0),
    )


def lower_convex_hull_mask(
    x,
    gibbs_energy,
    *,
    energy_scale: float | None = None,
    atol: float = 1.0e-10,
) -> np.ndarray:
    """Return a mask for points on the lower convex hull of G(x).

    The composition is represented by the first N-1 independent components.
    If there are too few points or the hull is degenerate, all finite points are
    kept as a conservative fallback.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(gibbs_energy, dtype=float).reshape(-1)
    finite = np.isfinite(g) & np.all(np.isfinite(x), axis=1)
    if finite.sum() <= x.shape[1]:
        return finite

    x_finite = x[finite]
    g_finite = g[finite]
    independent_x = x_finite[:, :-1]
    if independent_x.shape[1] == 0:
        mask = np.zeros_like(finite, dtype=bool)
        mask[np.flatnonzero(finite)[int(np.argmin(g_finite))]] = True
        return mask

    if energy_scale is None:
        energy_scale = float(np.nanstd(g_finite))
        if not np.isfinite(energy_scale) or energy_scale <= 0:
            energy_scale = 1.0

    points = np.column_stack([independent_x, g_finite / energy_scale])
    if points.shape[0] <= points.shape[1]:
        return finite

    try:
        from scipy.spatial import ConvexHull, QhullError

        hull = ConvexHull(points)
    except (ImportError, ValueError, QhullError):
        return finite

    local_mask = np.zeros(points.shape[0], dtype=bool)
    energy_normal_index = points.shape[1] - 1
    for simplex, equation in zip(hull.simplices, hull.equations):
        if equation[energy_normal_index] < -atol:
            local_mask[simplex] = True

    if not local_mask.any():
        return finite

    mask = np.zeros_like(finite, dtype=bool)
    mask[np.flatnonzero(finite)[local_mask]] = True
    return mask


def _temperature_powers(temperature: Tensor, polynomial_order: int, temperature_ref: float) -> Tensor:
    powers = torch.arange(
        polynomial_order + 1,
        device=temperature.device,
        dtype=temperature.dtype,
    )
    return (temperature[..., None] / temperature_ref) ** powers


def _ideal_mixing_gibbs(x: Tensor, temperature: Tensor) -> Tensor:
    x = x.clamp_min(1.0e-12)
    return R * temperature * (x * x.log()).sum(dim=-1)


def _redlich_kister_design_matrix(
    x: Tensor,
    temperature: Tensor,
    *,
    polynomial_order: int,
    interaction_order: int,
    temperature_ref: float,
) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    x = as_float_tensor(x, dtype=TORCH_FLOAT)
    temperature = as_float_tensor(temperature, dtype=x.dtype, device=x.device)
    n_points, n_components = x.shape
    t_powers = _temperature_powers(
        temperature,
        polynomial_order,
        temperature_ref,
    )

    columns = []
    labels: list[tuple[str, tuple[int, ...]]] = []

    for component_index in range(n_components):
        for temp_order in range(polynomial_order + 1):
            columns.append(x[:, component_index] * t_powers[:, temp_order])
            labels.append(("endmember", (component_index, temp_order)))

    for i in range(n_components):
        for j in range(i + 1, n_components):
            pair_factor = x[:, i] * x[:, j]
            delta = x[:, i] - x[:, j]
            for interaction_order_index in range(interaction_order + 1):
                composition_factor = pair_factor * delta.pow(interaction_order_index)
                for temp_order in range(polynomial_order + 1):
                    columns.append(composition_factor * t_powers[:, temp_order])
                    labels.append(
                        (
                            "interaction",
                            (i, j, interaction_order_index, temp_order),
                        )
                    )

    if not columns:
        return torch.empty((n_points, 0), dtype=x.dtype, device=x.device), labels
    return torch.stack(columns, dim=1), labels


def fit_redlich_kister_coefficients(
    x,
    temperature,
    gibbs_energy,
    *,
    elements: Sequence[str],
    name: str | None = None,
    polynomial_order: int = 1,
    interaction_order: int = 0,
    temperature_ref: float = 1000.0,
    ridge: float = 0.0,
    subtract_ideal_mixing: bool = True,
) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    """Fit CALPHAD solution parameters to sampled composition-energy data."""
    x = as_float_tensor(x, dtype=TORCH_FLOAT)
    temperature = as_float_tensor(temperature, dtype=x.dtype, device=x.device).reshape(-1)
    gibbs_energy = as_float_tensor(
        gibbs_energy,
        dtype=x.dtype,
        device=x.device,
    ).reshape(-1)
    if subtract_ideal_mixing:
        gibbs_energy = gibbs_energy - _ideal_mixing_gibbs(x, temperature)

    design, labels = _redlich_kister_design_matrix(
        x,
        temperature,
        polynomial_order=polynomial_order,
        interaction_order=interaction_order,
        temperature_ref=temperature_ref,
    )
    if design.shape[0] < design.shape[1] and ridge <= 0:
        ridge = 1.0e-10

    if ridge:
        lhs = design.T @ design + ridge * torch.eye(
            design.shape[1],
            dtype=design.dtype,
            device=design.device,
        )
        rhs = design.T @ gibbs_energy
        coeffs = torch.linalg.solve(lhs, rhs)
    else:
        coeffs = torch.linalg.lstsq(design, gibbs_energy).solution

    return coeffs.detach().cpu(), labels


def fit_compound_coefficients(
    temperature,
    gibbs_energy,
    *,
    polynomial_order: int = 1,
    temperature_ref: float = 1000.0,
    ridge: float = 0.0,
) -> Tensor:
    """Fit temperature-only compound parameters to sampled energy data."""
    temperature = as_float_tensor(temperature, dtype=TORCH_FLOAT).reshape(-1)
    gibbs_energy = as_float_tensor(gibbs_energy, dtype=temperature.dtype).reshape(-1)
    design = _temperature_powers(temperature, polynomial_order, temperature_ref)
    if design.shape[0] < design.shape[1] and ridge <= 0:
        ridge = 1.0e-10
    if ridge:
        lhs = design.T @ design + ridge * torch.eye(
            design.shape[1],
            dtype=design.dtype,
            device=design.device,
        )
        rhs = design.T @ gibbs_energy
        coeffs = torch.linalg.solve(lhs, rhs)
    else:
        coeffs = torch.linalg.lstsq(design, gibbs_energy).solution

    return coeffs.detach().cpu()


def _tdb_number(value: float, *, atol: float = 1.0e-12) -> str:
    value = float(value)
    if abs(value) < atol:
        value = 0.0
    return f"{value:.12g}"


def _tdb_polynomial(coeffs, temperature_ref: float) -> str:
    terms = []
    for order, coeff in enumerate(np.asarray(coeffs, dtype=float).reshape(-1)):
        if abs(coeff) < 1.0e-10:
            continue
        coefficient = _tdb_number(coeff)
        if order == 0:
            term = coefficient
        elif order == 1:
            term = f"{coefficient}*T/{_tdb_number(temperature_ref)}"
        else:
            term = f"{coefficient}*(T/{_tdb_number(temperature_ref)})**{order}"
        terms.append(term)
    if not terms:
        return "0"
    expression = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            expression += term
        else:
            expression += "+" + term
    return expression


def _render_solution_tdb(
    coeffs: Tensor,
    labels: list[tuple[str, tuple[int, ...]]],
    *,
    phase_name: str,
    elements: Sequence[str],
    polynomial_order: int,
    interaction_order: int,
    temperature_ref: float,
    t_min: float,
    t_max: float,
    include_phase_definition: bool,
) -> str:
    endmember = np.zeros((len(elements), polynomial_order + 1), dtype=float)
    interaction: dict[tuple[int, int, int], np.ndarray] = {}
    for value, (kind, indices) in zip(coeffs.numpy(), labels):
        if kind == "endmember":
            component_index, temp_order = indices
            endmember[component_index, temp_order] = float(value)
        else:
            i, j, interaction_order_index, temp_order = indices
            key = (i, j, interaction_order_index)
            interaction.setdefault(
                key,
                np.zeros(polynomial_order + 1, dtype=float),
            )[temp_order] = float(value)

    lines = []
    if include_phase_definition:
        lines.extend([
            f" PHASE {phase_name} %  1  1.0 !",
            f" CONSTITUENT {phase_name} :{','.join(elements)} : !",
        ])
    for element, coeff in zip(elements, endmember):
        lines.append(
            f" PARAMETER G({phase_name},{element};0) {t_min:g} "
            f"{_tdb_polynomial(coeff, temperature_ref)}; {t_max:g} N !"
        )
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            for order in range(interaction_order + 1):
                coeff = interaction.get(
                    (i, j, order),
                    np.zeros(polynomial_order + 1, dtype=float),
                )
                lines.append(
                    f" PARAMETER G({phase_name},{elements[i]},{elements[j]};{order}) "
                    f"{t_min:g} {_tdb_polynomial(coeff, temperature_ref)}; "
                    f"{t_max:g} N !"
                )
    return "\n".join(lines)


def _composition_to_site_ratios(x: np.ndarray) -> list[float]:
    fractions = [Fraction(float(value)).limit_denominator(16) for value in x]
    denominator_lcm = 1
    for fraction in fractions:
        denominator_lcm = np.lcm(denominator_lcm, fraction.denominator)
    ratios = [float(fraction.numerator * denominator_lcm // fraction.denominator) for fraction in fractions]
    if not any(ratios):
        return [float(value) for value in x]
    return ratios


def _render_compound_tdb(
    coeffs: Tensor,
    *,
    phase_name: str,
    elements: Sequence[str],
    composition: Sequence[float],
    temperature_ref: float,
    t_min: float,
    t_max: float,
    include_phase_definition: bool,
) -> str:
    lines = []
    if include_phase_definition:
        ratios = _composition_to_site_ratios(np.asarray(composition, dtype=float))
        sublattices = " ".join(_tdb_number(ratio) for ratio in ratios)
        constituents = " ".join(f": {element}" for element in elements)
        lines.extend([
            f" PHASE {phase_name} %  {len(elements)}  {sublattices} !",
            f" CONSTITUENT {phase_name} {constituents} : !",
        ])
    lines.append(
        f" PARAMETER G({phase_name},{':'.join(elements)};0) {t_min:g} "
        f"{_tdb_polynomial(coeffs.numpy(), temperature_ref)}; {t_max:g} N !"
    )
    return "\n".join(lines)


def export_corrected_model(
    model: GibbsModel,
    temperatures,
    *,
    n_samples_each_side: int = 16,
    polynomial_order: int = 1,
    interaction_order: int = 0,
    temperature_ref: float = 1000.0,
    use_lower_hull: bool = True,
    composition_atol: float = 1.0e-6,
    ridge: float = 0.0,
    name: str | None = None,
    include_phase_definition: bool = True,
) -> str:
    """Fit and render a TDB-style CALPHAD model string.

    Solution-like phases are rendered as a one-sublattice substitutional
    CALPHAD phase. Their fitted target subtracts ideal mixing, because
    pycalphad will add configurational entropy from the phase definition.
    Compound-like phases are rendered as stoichiometric compounds.
    """
    surface = sample_model_surface(
        model,
        temperatures,
        n_samples_each_side=n_samples_each_side,
    )
    x = surface.x.detach().cpu().numpy()
    g = surface.gibbs_energy.detach().cpu().numpy()
    t = surface.temperature.detach().cpu().numpy()

    composition_spread = np.max(x, axis=0) - np.min(x, axis=0)
    is_compound_like = np.all(composition_spread <= composition_atol)
    export_name = name or f"{model.phase_name}_export"
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if t_max <= t_min:
        t_max = t_min + 1.0

    if is_compound_like:
        coeffs = fit_compound_coefficients(
            t,
            g,
            polynomial_order=polynomial_order,
            temperature_ref=temperature_ref,
            ridge=ridge,
        )
        return _render_compound_tdb(
            coeffs,
            phase_name=export_name,
            elements=model.elements,
            composition=x.mean(axis=0),
            temperature_ref=temperature_ref,
            t_min=t_min,
            t_max=t_max,
            include_phase_definition=include_phase_definition,
        )

    if use_lower_hull:
        keep = np.zeros(x.shape[0], dtype=bool)
        for temperature in np.unique(t):
            temperature_mask = np.isclose(t, temperature)
            local_keep = lower_convex_hull_mask(
                x[temperature_mask],
                g[temperature_mask],
            )
            keep[np.flatnonzero(temperature_mask)[local_keep]] = True
        if keep.any():
            x = x[keep]
            g = g[keep]
            t = t[keep]

    coeffs, labels = fit_redlich_kister_coefficients(
        x,
        t,
        g,
        elements=model.elements,
        name=export_name,
        polynomial_order=polynomial_order,
        interaction_order=interaction_order,
        temperature_ref=temperature_ref,
        ridge=ridge,
    )
    return _render_solution_tdb(
        coeffs,
        labels,
        phase_name=export_name,
        elements=model.elements,
        polynomial_order=polynomial_order,
        interaction_order=interaction_order,
        temperature_ref=temperature_ref,
        t_min=t_min,
        t_max=t_max,
        include_phase_definition=include_phase_definition,
    )
