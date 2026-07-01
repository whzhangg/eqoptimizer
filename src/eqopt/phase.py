import dataclasses
from collections.abc import Set, Sequence, Mapping


@dataclasses.dataclass(frozen=True)
class PhaseID:
    """maybe captialize everything here?"""
    name: str
    elements: tuple[str, ...]
    prototype_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "elements", tuple(sorted(self.elements)))


    def __repr__(self):
        system = '-'.join(self.elements)
        postfix = f'({self.prototype_name})' if self.prototype_name else ''
        return f'{system}:{self.name}{postfix}'


def get_chemical_system(phases: Sequence[PhaseID]) -> Set[str]:
    ele = set()
    for phase in phases:
        ele.update(phase.elements)
    return ele
    

@dataclasses.dataclass
class PhaseEquilibrium:
    phases: Sequence[PhaseID]
    phase_compositions: Sequence[Mapping[str, float] | None] # ordered as phases
    temperature: float


    @property
    def chemical_system(self) -> Set[str]:
        return get_chemical_system(self.phases)
    

    def __repr__(self) -> str:
        s = '-'.join(tuple(sorted(get_chemical_system(self.phases))))
        s+= f': T = {self.temperature:g}, '
        parts = []
        for phase, composition in zip(self.phases, self.phase_compositions):
            if composition is None:
                p = f'{phase.name}(unknown)'
            else:
                sorted_ele = sorted(list(composition.keys()))
                p = f'{phase.name}('
                p+= ','.join([f'x_{ele}={composition[ele]:.3f}' for ele in sorted_ele])
                p+= ')'
            parts.append(p)
        
        s += ' = '.join(parts)
        return s
    

    def get_competing_phases(self, all_phases:Sequence[PhaseID]) -> Sequence[PhaseID]:
        elements_here = get_chemical_system(self.phases)
        return [ph for ph in all_phases if set(ph.elements)  <= elements_here]