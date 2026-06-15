import dataclasses
from collections.abc import Sequence, Mapping
import numpy as np

from .utilities import PRESSURE
from .phase import PhaseID, PhaseEquilibrium

VACANCY_COMPONENTS = {"VA", "/-"}


@dataclasses.dataclass
class PhaseModels:
    components: Sequence[str]
    phases: Mapping[str, Mapping]

    @classmethod
    def from_tdbfile(cls, tdbfilename: str):
        with open(tdbfilename, "r") as f:
            tdb_content = [line.strip().upper() for line in f if line.strip()]
        
        tdb_content = [line for line in tdb_content if not line.startswith('$')]
        all_elements = []
        all_phases = []
        phase_model = {}
        for aline in tdb_content:
            if aline.startswith("ELEMENT"):
                all_elements.append(aline.split()[1])
            if aline.startswith("PHASE"):
                all_phases.append(aline.split()[1])
        phase_model["components"] = all_elements
        phase_model["phases"] = {}
        for phase in all_phases:
            sublattices = None
            sublattice_constituent = None
            for aline in tdb_content:
                splitted = aline.split()
                if "PHASE" in splitted and phase in splitted:
                    nsub = int(splitted[3])
                    sublattices = [float(p) for p in splitted[4 : 4 + nsub]]
                    break
            for aline in tdb_content:
                splitted = aline.split()
                if "CONSTITUENT" in splitted and phase in splitted and sublattices is not None:
                    nsub = len(sublattices)
                    sublattice_constituent = [
                        [pp.strip() for pp in p.split(",")]
                        for p in aline.split(":")[1 : 1 + nsub]
                    ]
                    break
            if sublattices is not None and sublattice_constituent is not None:
                phase_model["phases"][phase] = {
                    "sublattice_model": sublattice_constituent,
                    "sublattice_site_ratios": sublattices,
                }

        if '/-' in phase_model['components']:
            phase_model['components'].remove('/-')
        return cls(
            phase_model['components'],
            phase_model['phases']
        )


    def get_phase_elements(self, name: str) -> tuple[str, ...]:
        """Return real elements allowed on the phase constituents."""
        if name not in self.phases:
            return tuple(
                component for component in self.components
                if component not in VACANCY_COMPONENTS
            )
        elements = set()
        for sublattice in self.phases[name]["sublattice_model"]:
            for component in sublattice:
                if component not in VACANCY_COMPONENTS:
                    elements.add(component)
        return tuple(sorted(elements))


    def get_composition_if_stoichmetric(self, name) -> Mapping[str, float] | None:
        if name not in self.phases:
            return None
        model = self.phases[name]
        is_compound = True
        for eles in model["sublattice_model"]:
            if len(eles) > 1:
                is_compound = False
                break
        if is_compound:
            real_site_ratios = [
                ratio
                for ele, ratio in zip(
                    model["sublattice_model"], model["sublattice_site_ratios"]
                )
                if ele[0] not in VACANCY_COMPONENTS
            ]
            ntot = sum(real_site_ratios)
            return {
                ele[0]: ratio / ntot
                for ele, ratio in zip(
                    model["sublattice_model"], model["sublattice_site_ratios"]
                )
                if ele[0] not in VACANCY_COMPONENTS
            }
        else:
            return None


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
            _composition_key(
                phase_name,
                composition,
                components,
                composition_atol,
            )
            for phase_name, composition in phase_compositions
        )
    )


def _composition_from_xarray(
    x_values: np.ndarray,
    components: Sequence[str],
) -> Mapping[str, float]:
    x_values = np.asarray(x_values, dtype=float)
    return {
        component: float(value)
        for component, value in zip(components, x_values)
        if np.isfinite(value)
    }


class TDBHandler:
    """
    A general handler of TDB files, and that can interact with VVxxxx
    parameters
    """

    def __init__(self, tdbfilename: str):
        self.tdbfilename = tdbfilename
        self.phase_models = PhaseModels.from_tdbfile(tdbfilename)
        

    @property
    def components(self) -> list[str]:
        return list(self.phase_models.components)


    @property
    def phase_names(self) -> list[str]:
        return list(self.phase_models.phases.keys())


    def get_phase_ids(self) -> Sequence[PhaseID]:
        """find all phaseID for all phases defined"""
        ids = []
        for phase in self.phase_models.phases:
            ids.append(PhaseID(phase, self.phase_models.get_phase_elements(phase)))
        return ids


    def build_equilibrium_data(
        self,
        temperature: float,
        phase_models: PhaseModels | None = None, 
        nsamples: int = 64,
        composition_atol: float = 1.0e-5,
        phase_fraction_atol: float = 1.0e-8,
        multiphase_only: bool = True
    ) -> Sequence[PhaseEquilibrium]:
        """Calculate and deduplicate observed phase compositions at a temperature.

        Phase fractions are used only to decide whether a vertex is present. The
        uniqueness key is the sorted set of phase names and phase compositions.
        """
        from pycalphad import equilibrium, variables as v
        phase_model_to_use = phase_models or self.phase_models
        components = [c for c in self.components if c not in VACANCY_COMPONENTS]
        
        conditions = {
            v.P: PRESSURE, 
            v.T: temperature, 
        }
        composition_grid = np.linspace(0.0, 1.0, nsamples)
        for ic in range(len(components)-1):
            conditions[v.X(components[ic])] = composition_grid
        
        eq = equilibrium(
            self.tdbfilename, comps=self.components, phases=self.phase_names, 
            conditions=conditions
        ) 
        
        found_equilibrium = []
        found_keys = set()

        phase_values = eq.Phase.values
        #print(phase_values)
        phase_fraction_values = eq.NP.values
        composition_values = eq.X.sel(component=components).values
        vertex_count = phase_values.shape[-1]
        grid_shape = phase_values.shape[:-1]
        #print(vertex_count)
        #print(grid_shape)

        for grid_index in np.ndindex(grid_shape):
            phase_compositions = []
            for vertex_index in range(vertex_count):
                phase_fraction = phase_fraction_values[grid_index + (vertex_index,)]
                if (
                    not np.isfinite(phase_fraction)
                    or phase_fraction <= phase_fraction_atol
                ):
                    continue

                phase_name = str(phase_values[grid_index + (vertex_index,)]).strip()
                if not phase_name:
                    continue

                stoichiometric = phase_model_to_use.get_composition_if_stoichmetric(
                    phase_name
                )
                if stoichiometric is not None:
                    phase_composition = stoichiometric
                else:
                    phase_composition = _composition_from_xarray(
                            composition_values[grid_index + (vertex_index, slice(None))],
                            components,
                    )
                phase_compositions.append((phase_name, phase_composition))
            #print(phase_compositions)
            if not phase_compositions:
                continue

            if multiphase_only and len(phase_compositions) == 1:
                continue

            phase_compositions = list(
                {
                    _composition_key(
                        phase_name,
                        composition,
                        components,
                        composition_atol,
                    ): (phase_name, composition)
                    for phase_name, composition in phase_compositions
                }.values()
            )

            key = _equilibrium_key(
                phase_compositions,
                components,
                composition_atol,
            )
            if key in found_keys:
                continue

            found_keys.add(key)
            sorted_phase_compositions = sorted(
                phase_compositions,
                key=lambda phase_composition: _composition_key(
                    phase_composition[0],
                    phase_composition[1],
                    components,
                    composition_atol,
                ),
            )
            found_equilibrium.append(
                PhaseEquilibrium(
                    phases=[
                        PhaseID(
                            name=phase_name,
                            elements=phase_model_to_use.get_phase_elements(phase_name),
                        )
                        for phase_name, _ in sorted_phase_compositions
                    ],
                    phase_compositions=[
                        dict(composition)
                        for _, composition in sorted_phase_compositions
                    ],
                    temperature=temperature,
                )
            )

        return found_equilibrium
