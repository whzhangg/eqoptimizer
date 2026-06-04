import dataclasses
import typing
import numpy as np

from .utilities import simplex_samples_dirichlet, PRESSURE

@dataclasses.dataclass
class PhaseCompositions:
    """specify a phase with a composition"""
    name: str
    compositions: typing.Dict[str, float]

    def __repr__(self):
        sorted_ele = sorted(list(self.compositions.keys()))
        s = f'{self.name}('
        s+= ','.join([f'x_{ele}={self.compositions[ele]:.3f}' for ele in sorted_ele])
        return s + ')'


@dataclasses.dataclass
class EquilibriumCompositions:
    """Observed phase compositions for one distinct equilibrium."""
    temperature: float
    phases: typing.List[PhaseCompositions]

    def __repr__(self):
        s = f'T = {self.temperature:g} '
        s += ' = '.join(str(phase) for phase in self.phases)
        return s


@dataclasses.dataclass
class PhaseModels:
    components: typing.List[str]
    phases: typing.Dict[str, typing.Any]

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


    def get_composition_if_stoichmetric(self, name) -> typing.Optional[PhaseCompositions]:
        if name not in self.phases:
            return None
        model = self.phases[name]
        is_compound = True
        for eles in model["sublattice_model"]:
            if len(eles) > 1:
                is_compound = False
                break
        if is_compound:
            ntot = sum(model["sublattice_site_ratios"])
            return PhaseCompositions(
                    name=name,
                    compositions={
                        ele[0]: ratio / ntot
                        for ele, ratio in zip(
                            model["sublattice_model"], model["sublattice_site_ratios"]
                        )
                    }
                )
        else:
            return None


def _composition_key(
    phase_composition: PhaseCompositions,
    components: typing.Sequence[str],
    composition_atol: float,
) -> tuple[str, tuple[int, ...]]:
    scale = 1.0 / composition_atol
    return (
        phase_composition.name,
        tuple(
            int(round(phase_composition.compositions.get(component, 0.0) * scale))
            for component in components
        ),
    )


def _equilibrium_key(
    phase_compositions: typing.Sequence[PhaseCompositions],
    components: typing.Sequence[str],
    composition_atol: float,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            _composition_key(phase_composition, components, composition_atol)
            for phase_composition in phase_compositions
        )
    )


def _composition_from_xarray(
    x_values: np.ndarray,
    components: typing.Sequence[str],
) -> typing.Dict[str, float]:
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


    def build_equilibrium_data(
        self,
        temperature: float,
        phase_models: typing.Optional[PhaseModels] = None, 
        nsamples: int = 64,
        composition_atol: float = 1.0e-5,
        phase_fraction_atol: float = 1.0e-8,
        multiphase_only: bool = True
    ) -> typing.List[EquilibriumCompositions]:
        """Calculate and deduplicate observed phase compositions at a temperature.

        Phase fractions are used only to decide whether a vertex is present. The
        uniqueness key is the sorted set of phase names and phase compositions.
        """
        from pycalphad import equilibrium, variables as v
        phase_model_to_use = phase_models or self.phase_models
        components = [c for c in self.components if c not in ('VA', '/-')]
        
        sampled = simplex_samples_dirichlet(
            n_components=len(components),
            n_samples_each_side=nsamples    
        ).detach().numpy()

        conditions = {
            v.P: PRESSURE, 
            v.T: temperature, 
        }
        for ic in range(len(components)-1):
            conditions[v.X(components[ic])] = sampled[:,ic]
        
        #print(conditions)
        #print(self.phase_names)
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
                    phase_composition = PhaseCompositions(
                        name=phase_name,
                        compositions=_composition_from_xarray(
                            composition_values[grid_index + (vertex_index, slice(None))],
                            components,
                        ),
                    )
                phase_compositions.append(phase_composition)
            #print(phase_compositions)
            if not phase_compositions:
                continue

            if multiphase_only and len(phase_compositions) == 1:
                continue

            phase_compositions = list(
                {
                    _composition_key(
                        phase_composition,
                        components,
                        composition_atol,
                    ): phase_composition
                    for phase_composition in phase_compositions
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
            found_equilibrium.append(
                EquilibriumCompositions(
                    temperature=temperature,
                    phases=sorted(
                        phase_compositions,
                        key=lambda phase_composition: _composition_key(
                            phase_composition,
                            components,
                            composition_atol,
                        ),
                    ),
                )
            )

        return found_equilibrium


if __name__ == '__main__':
    import pathlib

    ref = pathlib.Path(__file__).resolve().parents[2] / "examples" / "CPDDB_PbSn.tdb"
    handler = TDBHandler(str(ref))
    print(handler.build_equilibrium_data(550))
