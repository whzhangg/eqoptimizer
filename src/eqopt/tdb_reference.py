from collections.abc import Mapping, Sequence

import numpy as np

from .phase import PhaseEquilibrium, PhaseID
from .utilities import PRESSURE


__all__ = ["TDBHandler"]


_VACANCY_COMPONENTS = {"VA", "/-"}


class TDBHandler:
    """Generate phase identifiers and equilibrium data from a TDB file."""

    def __init__(self, tdbfilename: str):
        from pycalphad import Database

        self._database = Database(tdbfilename)
        self._components = tuple(
            component
            for component in sorted(self._database.elements)
            if component != "/-"
        )
        self._real_components = tuple(
            component
            for component in self._components
            if component not in _VACANCY_COMPONENTS
        )


    def get_phase_ids(self) -> Sequence[PhaseID]:
        return [
            PhaseID(phase_name, _phase_elements(phase))
            for phase_name, phase in self._database.phases.items()
        ]


    def build_equilibrium_data(
        self,
        temperature: float,
        nsamples: int = 64,
        composition_atol: float = 1.0e-3,
        phase_fraction_atol: float = 1.0e-3,
        multiphase_only: bool = True,
    ) -> Sequence[PhaseEquilibrium]:
        """Calculate and deduplicate observed phase compositions at one temperature."""
        from pycalphad import equilibrium, variables as v

        conditions = {v.P: PRESSURE, v.T: temperature}
        composition_grid = np.linspace(0.0, 1.0, nsamples)

        if len(self._real_components) <= 1:
            independent_component_sets = [tuple()]
        else:
            independent_component_sets = [
                tuple(
                    component
                    for component in self._real_components
                    if component != dependent_component
                )
                for dependent_component in self._real_components
            ]

        found_equilibria = []
        found_keys = set()

        # Repeating with each dependent component recovers boundary points that pycalphad
        # can omit when only one choice of independent components is used.
        for independent_components in independent_component_sets:
            local_conditions = dict(conditions)
            for component in independent_components:
                local_conditions[v.X(component)] = composition_grid

            eq = equilibrium(
                self._database,
                comps=self._components,
                phases=tuple(self._database.phases),
                conditions=local_conditions,
            )
            phase_values = eq.Phase.values
            phase_fraction_values = eq.NP.values
            composition_values = eq.X.sel(component=list(self._real_components)).values
            vertex_count = phase_values.shape[-1]

            for grid_index in np.ndindex(phase_values.shape[:-1]):
                phase_compositions = []
                for vertex_index in range(vertex_count):
                    index = grid_index + (vertex_index,)
                    phase_fraction = phase_fraction_values[index]
                    if not np.isfinite(phase_fraction) or phase_fraction <= phase_fraction_atol:
                        continue

                    phase_name = str(phase_values[index]).strip()
                    if not phase_name:
                        continue

                    phase = self._database.phases[phase_name]
                    phase_composition = _stoichiometric_composition(phase)
                    if phase_composition is None:
                        phase_composition = _composition_from_xarray(
                            composition_values[index + (slice(None),)],
                            self._real_components,
                        )
                    phase_compositions.append((phase_name, phase_composition))

                if not phase_compositions:
                    continue
                if multiphase_only and len(phase_compositions) == 1:
                    continue

                phase_compositions = list(
                    {
                        _composition_key(
                            phase_name,
                            composition,
                            self._real_components,
                            composition_atol,
                        ): (phase_name, composition)
                        for phase_name, composition in phase_compositions
                    }.values()
                )
                key = _equilibrium_key(
                    phase_compositions,
                    self._real_components,
                    composition_atol,
                )
                if key in found_keys:
                    continue
                found_keys.add(key)

                phase_compositions.sort(
                    key=lambda item: _composition_key(
                        item[0],
                        item[1],
                        self._real_components,
                        composition_atol,
                    )
                )
                found_equilibria.append(
                    PhaseEquilibrium(
                        phases=[
                            PhaseID(
                                name=phase_name,
                                elements=_phase_elements(self._database.phases[phase_name]),
                            )
                            for phase_name, _ in phase_compositions
                        ],
                        phase_compositions=[
                            dict(composition)
                            for _, composition in phase_compositions
                        ],
                        temperature=temperature,
                    )
                )

        return found_equilibria


def _phase_elements(phase) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                species.name
                for sublattice in phase.constituents
                for species in sublattice
                if species.name not in _VACANCY_COMPONENTS
            }
        )
    )


def _stoichiometric_composition(phase) -> Mapping[str, float] | None:
    if any(len(sublattice) > 1 for sublattice in phase.constituents):
        return None

    real_sublattices = [
        (next(iter(sublattice)).name, float(ratio))
        for sublattice, ratio in zip(phase.constituents, phase.sublattices)
        if next(iter(sublattice)).name not in _VACANCY_COMPONENTS
    ]
    total = sum(ratio for _, ratio in real_sublattices)
    return {
        element: sum(ratio for name, ratio in real_sublattices if name == element) / total
        for element, _ in real_sublattices
    }


def _composition_key(
    phase_name: str,
    composition: Mapping[str, float],
    components: Sequence[str],
    composition_atol: float,
) -> tuple[str, tuple[int, ...]]:
    scale = 1.0 / composition_atol
    return (
        phase_name,
        tuple(
            int(round(composition.get(component, 0.0) * scale))
            for component in components
        ),
    )


def _equilibrium_key(
    phase_compositions: Sequence[tuple[str, Mapping[str, float]]],
    components: Sequence[str],
    composition_atol: float,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            _composition_key(phase_name, composition, components, composition_atol)
            for phase_name, composition in phase_compositions
        )
    )


def _composition_from_xarray(
    x_values: np.ndarray,
    components: Sequence[str],
) -> Mapping[str, float]:
    return {
        component: float(value)
        for component, value in zip(components, np.asarray(x_values, dtype=float))
        if np.isfinite(value)
    }
