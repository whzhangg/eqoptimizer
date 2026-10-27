from abc import ABC, abstractmethod
import dataclasses
from collections.abc import Mapping, Sequence
import torch

from .polynomial import TemperatureExpression


@dataclasses.dataclass(frozen=True)
class CEFContext:
    """Coordinate context needed by CEF energy terms."""
    y_names_to_index: Mapping[tuple[str, int], int]
    sublattice_multiplicities: tuple[float, ...]
    phase_name: str | None = None

    def site_fraction(
        self,
        y: torch.Tensor,
        component: str,
        sublattice: int,
    ) -> torch.Tensor:
        return y[..., self.y_names_to_index[(component, sublattice)]]

    def conditioner_product(
        self,
        y: torch.Tensor,
        conditioners: Sequence[tuple[int, str]],
    ) -> torch.Tensor:
        indices = [
            self.y_names_to_index[(component, sublattice)]
            for sublattice, component in conditioners
            if component != "*"
        ]
        if not indices:
            return torch.ones(y.shape[:-1], device=y.device, dtype=y.dtype)
        indices = torch.as_tensor(
            indices,
            device=y.device,
            dtype=torch.int64,
        )
        return y[..., indices].prod(dim=-1)

    def sublattice_multiplicity(self, sublattice: int) -> float:
        return self.sublattice_multiplicities[sublattice]

    @property
    def nsublattice(self) -> int:
        return len(self.sublattice_multiplicities)

    def require_site_fraction(self, component: str, sublattice: int) -> None:
        if component == "*":
            return
        if (component, sublattice) not in self.y_names_to_index:
            raise ValueError(
                f"Unknown site fraction ({component!r}, {sublattice}); "
                f"available site fractions are {tuple(self.y_names_to_index)}."
            )

    def require_complete_tdb_site_array(
        self,
        conditioners: Sequence[tuple[int, str]],
        mixed_sublattice: int | None = None,
    ) -> None:
        occupied_sublattices = [sublattice for sublattice, _ in conditioners]
        if mixed_sublattice is not None:
            occupied_sublattices.append(mixed_sublattice)

        duplicate_sublattices = {
            sublattice
            for sublattice in occupied_sublattices
            if occupied_sublattices.count(sublattice) > 1
        }
        if duplicate_sublattices:
            raise ValueError(
                f"Multiple entries for sublattices "
                f"{sorted(duplicate_sublattices)}."
            )

        missing = [
            sublattice
            for sublattice in range(self.nsublattice)
            if sublattice not in occupied_sublattices
        ]
        if missing:
            raise ValueError(f"Missing entries for sublattices {missing}.")

    def tdb_site_array(
        self,
        conditioners: Sequence[tuple[int, str]],
        mixed_sublattice: int | None = None,
        mixed_components: Sequence[str] | None = None,
    ) -> str:
        sites_symbol = {
            sublattice: component
            for sublattice, component in conditioners
        }
        if mixed_sublattice is not None:
            if mixed_components is None:
                raise ValueError("mixed_components is required for a mixed sublattice.")
            sites_symbol[mixed_sublattice] = ",".join(mixed_components)
        return ":".join(sites_symbol[i] for i in range(self.nsublattice))

    def tdb_site_array_from_entries(
        self,
        entries: Mapping[int, str | Sequence[str]],
    ) -> str:
        missing = [
            sublattice
            for sublattice in range(self.nsublattice)
            if sublattice not in entries
        ]
        if missing:
            raise ValueError(f"Missing entries for sublattices {missing}.")
        parts = []
        for sublattice in range(self.nsublattice):
            entry = entries[sublattice]
            if isinstance(entry, str):
                parts.append(entry)
            else:
                parts.append(",".join(entry))
        return ":".join(parts)


class CEFExcessTerm(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def get_contribution(
        self,
        y: torch.Tensor,
        temperature: torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        """get its contribution to the Gibbs energy"""

    def energy(
        self,
        y: torch.Tensor,
        temperature: torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        return self.get_contribution(y, temperature, context)

    @abstractmethod
    def to_tdb_str(self, context: CEFContext, parameter_name: str = "G") -> str:
        """parse itself to a TDB string"""


    @abstractmethod
    def validate(self, context: CEFContext) -> bool:
        """validate if self is consistent with context"""


def _tdb_parameter_str(
    phase_name: str,
    site_array: str,
    order: int,
    interaction: TemperatureExpression,
    parameter_name: str = "G",
) -> str:
    return (
        f"parameter {parameter_name}({phase_name},{site_array};{order}) "
        f"10.0  {interaction.get_expression()} ; 6000 N !"
    )


@dataclasses.dataclass
class BinaryExcessTerm(CEFExcessTerm):
    """prod y_condition * y_i * y_j * L(T) * (y_i - y_j)^order."""
    sublattice: int
    pair: tuple[str, str]
    order: int
    conditioners: tuple[tuple[int, str], ...]
    interaction: TemperatureExpression


    def get_contribution(
        self,
        y: torch.Tensor,
        temperature: torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        conditioner = context.conditioner_product(y, self.conditioners)
        yi = context.site_fraction(y, self.pair[0], self.sublattice)
        yj = context.site_fraction(y, self.pair[1], self.sublattice)
        coefficient = (
            conditioner
            * yi
            * yj
            * (yi - yj) ** self.order
        )
        return coefficient * self.interaction(temperature)

    
    def validate(self, context: CEFContext) -> bool:
        if self.order < 0:
            raise ValueError("BinaryExcessTerm order must be non-negative.")
        for sublattice, component in self.conditioners:
            context.require_site_fraction(component, sublattice)
        for component in self.pair:
            context.require_site_fraction(component, self.sublattice)
        context.require_complete_tdb_site_array(
            self.conditioners,
            mixed_sublattice=self.sublattice,
        )
        return True

    
    def to_tdb_str(self, context: CEFContext, parameter_name: str = "G") -> str:
        if context.phase_name is None:
            raise ValueError("CEFContext.phase_name is required for TDB export.")
        self.validate(context)
        site_array = context.tdb_site_array(
            self.conditioners,
            mixed_sublattice=self.sublattice,
            mixed_components=self.pair,
        )
        return _tdb_parameter_str(
            context.phase_name,
            site_array,
            self.order,
            self.interaction,
            parameter_name,
        )


@dataclasses.dataclass
class TernaryExcessTerm(CEFExcessTerm):
    """prod y_condition * y_i * y_j * y_k * v_order * L(T)."""
    sublattice: int
    triplet: tuple[str, str, str]
    order: int
    conditioners: tuple[tuple[int, str], ...]
    interaction: TemperatureExpression
    is_l0_only: bool


    def get_contribution(
        self,
        y: torch.Tensor,
        temperature: torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        conditioner = context.conditioner_product(y, self.conditioners)
        triplet_fractions = tuple(
            context.site_fraction(y, component, self.sublattice)
            for component in self.triplet
        )
        coefficient = conditioner
        for site_fraction in triplet_fractions:
            coefficient = coefficient * site_fraction
        
        if self.is_l0_only:
            """if only L0 is given, it is treated as L2=L1=L0"""
            return coefficient * self.interaction(temperature)
        else:
            triplet_sum = torch.stack(triplet_fractions, dim=0).sum(dim=0)
            v = triplet_fractions[self.order] + (1.0 - triplet_sum) / 3.0
            return coefficient * self.interaction(temperature) * v


    def validate(self, context: CEFContext) -> bool:
        if self.order < 0 or self.order > 2:
            raise ValueError("TernaryExcessTerm order must be 0, 1, or 2.")
        for sublattice, component in self.conditioners:
            context.require_site_fraction(component, sublattice)
        for component in self.triplet:
            context.require_site_fraction(component, self.sublattice)
        context.require_complete_tdb_site_array(
            self.conditioners,
            mixed_sublattice=self.sublattice,
        )
        return True


    def to_tdb_str(self, context: CEFContext, parameter_name: str = "G") -> str:
        if context.phase_name is None:
            raise ValueError("CEFContext.phase_name is required for TDB export.")
        self.validate(context)
        site_array = context.tdb_site_array(
            self.conditioners,
            mixed_sublattice=self.sublattice,
            mixed_components=self.triplet,
        )
        return _tdb_parameter_str(
            context.phase_name,
            site_array,
            self.order,
            self.interaction,
            parameter_name,
        )


@dataclasses.dataclass
class TwoSublatticeBinaryExcessTerm(CEFExcessTerm):
    sublattice: tuple[int, int]
    pair1: tuple[str, str]
    pair2: tuple[str, str]
    order: int
    conditioners: tuple[tuple[int, str], ...]
    interaction: TemperatureExpression

    def get_contribution(self, y, temperature, context):
        conditioner = context.conditioner_product(y, self.conditioners)
        y_i1 = context.site_fraction(y, self.pair1[0], self.sublattice[0])
        y_j1 = context.site_fraction(y, self.pair1[1], self.sublattice[0])
        y_m2 = context.site_fraction(y, self.pair2[0], self.sublattice[1])
        y_n2 = context.site_fraction(y, self.pair2[1], self.sublattice[1])
        coefficient = conditioner * y_i1 * y_j1 * y_m2 * y_n2
        if self.order == 0:
            return coefficient * self.interaction(temperature)
        elif self.order == 1:
            return coefficient * self.interaction(temperature) * (y_m2 - y_n2)
        elif self.order == 2:
            return coefficient * self.interaction(temperature) * (y_i1 - y_j1)
        else:
            raise ValueError('interaction order larger than 2')

    def validate(self, context: CEFContext) -> bool:
        if self.order < 0 or self.order > 2:
            raise ValueError(
                "TwoSublatticeBinaryExcessTerm order must be 0, 1, or 2."
            )
        if len(self.sublattice) != 2:
            raise ValueError("TwoSublatticeBinaryExcessTerm needs two sublattices.")
        if self.sublattice[0] == self.sublattice[1]:
            raise ValueError(
                "TwoSublatticeBinaryExcessTerm sublattices must be distinct."
            )
        if len(self.pair1) != 2 or len(self.pair2) != 2:
            raise ValueError("TwoSublatticeBinaryExcessTerm pairs must be binary.")
        for sublattice, component in self.conditioners:
            context.require_site_fraction(component, sublattice)
        for component in self.pair1:
            context.require_site_fraction(component, self.sublattice[0])
        for component in self.pair2:
            context.require_site_fraction(component, self.sublattice[1])
        occupied = [sublattice for sublattice, _ in self.conditioners]
        occupied.extend(self.sublattice)
        duplicate_sublattices = {
            sublattice
            for sublattice in occupied
            if occupied.count(sublattice) > 1
        }
        if duplicate_sublattices:
            raise ValueError(
                "TwoSublatticeBinaryExcessTerm has multiple entries for "
                f"sublattices {sorted(duplicate_sublattices)}."
            )
        missing = [
            sublattice
            for sublattice in range(context.nsublattice)
            if sublattice not in occupied
        ]
        if missing:
            raise ValueError(
                "TwoSublatticeBinaryExcessTerm is missing entries for "
                f"sublattices {missing}."
            )
        return True

    def to_tdb_str(self, context: CEFContext, parameter_name: str = "G") -> str:
        if context.phase_name is None:
            raise ValueError("CEFContext.phase_name is required for TDB export.")
        self.validate(context)
        entries: dict[int, str | Sequence[str]] = {
            sublattice: component
            for sublattice, component in self.conditioners
        }
        entries[self.sublattice[0]] = self.pair1
        entries[self.sublattice[1]] = self.pair2
        site_array = context.tdb_site_array_from_entries(entries)
        return _tdb_parameter_str(
            context.phase_name,
            site_array,
            self.order,
            self.interaction,
            parameter_name,
        )


@dataclasses.dataclass
class EndMemberTerm(CEFExcessTerm):
    """prod y_{condition} * interaction"""
    conditioners: tuple[tuple[int, str], ...]
    interaction: TemperatureExpression


    def get_contribution(
        self,
        y: torch.Tensor,
        temperature: torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        return context.conditioner_product(y, self.conditioners) * self.interaction(temperature)

    def validate(self, context: CEFContext) -> bool:
        for sublattice, component in self.conditioners:
            context.require_site_fraction(component, sublattice)
        context.require_complete_tdb_site_array(self.conditioners)
        return True

    def to_tdb_str(self, context: CEFContext, parameter_name: str = "G") -> str:
        if context.phase_name is None:
            raise ValueError("CEFContext.phase_name is required for TDB export.")
        self.validate(context)
        site_array = context.tdb_site_array(self.conditioners)
        return _tdb_parameter_str(
            context.phase_name,
            site_array,
            0,
            self.interaction,
            parameter_name,
        )
