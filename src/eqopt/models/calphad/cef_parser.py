from collections.abc import Callable, Sequence
from pathlib import Path
import re
from typing import TYPE_CHECKING

from .cef_surface import CEFContribution, CEFSurface
from .cef_terms import (
    BinaryExcessTerm,
    CEFContext,
    CEFExcessTerm,
    EndMemberTerm,
    TernaryExcessTerm,
    TwoSublatticeBinaryExcessTerm,
)
from .magnetic import CEFMagneticContribution
from .polynomial import (
    ParameterizedExpression,
    TemperatureExpression,
    TempPolynomial,
    TempPolynomialwCorrection,
)

if TYPE_CHECKING:
    from .cef_model import CEF, CEFConfig


_CORRECTION_PHYSICAL_SCALES = {
    "G": 5.0e3,
    "TC": 1.0e3,
    "BM": 1.0,
}


def _strip_tdb_comments(text: str) -> str:
    return re.sub(r'^\s*\$.*$', '', text, flags=re.MULTILINE)


def _tdb_commands(text: str) -> list[str]:
    return [
        command.strip()
        for command in _strip_tdb_comments(text).split('!')
        if command.strip()
    ]


def _normalize_tdb_sublattice_entry(entry: str) -> tuple[str, ...]:
    """Match pycalphad's TDB normalization: uppercase and sort per sublattice."""
    return tuple(
        sorted(
            component.strip().upper()
            for component in entry.split(',')
            if component.strip()
        )
    )


def _site_entries_to_string(site_entries: Sequence[Sequence[str]]) -> str:
    return ":".join(",".join(entry) for entry in site_entries)


def _parse_tdb_parameter_command(
    command: str,
    parameter_name: str,
) -> tuple[str, str, int, str] | None:
    pattern = (
        rf'PARA(?:METER)?\s+{re.escape(parameter_name)}\(\s*'
        r'([^,\s]+)\s*,\s*'
        r'(.+?)'
        r';\s*(\d+)\s*\)\s*'
        r'[-+0-9.Ee]+\s+'
        r'(.+?)'
        r';\s*[-+0-9.Ee]+\s+[YN]\s*$'
    )
    match = re.match(pattern, command.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    site_entries = tuple(
        _normalize_tdb_sublattice_entry(entry)
        for entry in match.group(2).split(':')
    )
    return (
        match.group(1).strip().upper(),
        _site_entries_to_string(site_entries),
        int(match.group(3)),
        match.group(4),
    )


def _get_cef_context_and_type_symbol(
    tdb_string: str,
    phase_name: str,
) -> tuple[CEFContext, str | None]:
    phase_name = phase_name.upper()
    multiplicity = None
    components = None
    type_symbol = None

    for command in _tdb_commands(tdb_string):
        upper_command = command.upper()
        if upper_command.startswith('PHASE'):
            tokens = upper_command.split()
            if len(tokens) >= 4 and tokens[1] == phase_name:
                symbol = tokens[2]
                type_symbol = symbol[1:] if symbol.startswith('%') and len(symbol) > 1 else None
                n_sublattices = int(tokens[3])
                multiplicity = tuple(
                    float(value)
                    for value in tokens[4:4 + n_sublattices]
                )

        if upper_command.startswith('CONSTITUENT'):
            match = re.match(
                r'CONSTITUENT\s+(\s*\S+\s*)\s*:(.*):\s*$',
                upper_command,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match and match.group(1).strip().upper() == phase_name:
                components = tuple(
                    _normalize_tdb_sublattice_entry(sublattice)
                    for sublattice in match.group(2).split(':')
                )

    if multiplicity is None or components is None:
        raise ValueError(f"Phase {phase_name!r} cannot be found or is incomplete.")
    if len(components) != len(multiplicity):
        raise ValueError(
            f"Phase {phase_name!r} has {len(components)} constituent "
            f"sublattices but {len(multiplicity)} multiplicities."
        )

    y_names_to_index = {}
    for sublattice, sublattice_components in enumerate(components):
        for component in sublattice_components:
            y_names_to_index[(component, sublattice)] = len(y_names_to_index)
    context = CEFContext(
        y_names_to_index=y_names_to_index,
        sublattice_multiplicities=multiplicity,
        phase_name=phase_name,
    )
    return context, type_symbol


def get_cef_context_from_tdb_string(tdb_string: str, phase_name: str) -> CEFContext:
    context, _ = _get_cef_context_and_type_symbol(tdb_string, phase_name)
    return context


def get_excess_term_from_tdb_string(
    command: str,
    context: CEFContext,
    *,
    parameter_name: str = "G",
    temperature_ref: float = 1000,
    correction_order: int | None = None,
    initial_parameter_scale: float = 0.1,
    ternary_l0_only: bool = False,
    interaction_factory: Callable[[str], TemperatureExpression] | None = None,
) -> CEFExcessTerm | None:
    parsed = _parse_tdb_parameter_command(command, parameter_name)
    if parsed is None:
        return None

    phase_name, site_array, order, expression = parsed
    if context.phase_name is not None and phase_name != context.phase_name.upper():
        return None

    site_entries = site_array.split(':')
    if len(site_entries) != context.nsublattice:
        raise ValueError(
            f"Expected {context.nsublattice} sublattice entries, "
            f"got {len(site_entries)}: {command}"
        )

    if interaction_factory is not None:
        interaction = interaction_factory(expression)
    elif correction_order is not None:
        parameter_name = parameter_name.upper()
        interaction = TempPolynomialwCorrection.from_expression(
            expression,
            list(range(correction_order + 1)),
            temperature_ref=temperature_ref,
            physical_scale=_CORRECTION_PHYSICAL_SCALES[parameter_name],
            initial_parameter_scale=initial_parameter_scale,
        )
    else:
        interaction = TempPolynomial.from_expression(
            expression,
            temperature_ref=temperature_ref,
        )

    mixed_sublattices = [
        index
        for index, entry in enumerate(site_entries)
        if ',' in entry
    ]
    if not mixed_sublattices:
        term = EndMemberTerm(
            conditioners=tuple(
                (index, entry)
                for index, entry in enumerate(site_entries)
            ),
            interaction=interaction,
        )
        term.validate(context)
        return term
    if len(mixed_sublattices) > 2:
        raise ValueError(
            f"At most two mixed sublattices are supported in restricted CEF "
            f"TDB import: {command}"
        )

    mixed_components_by_sublattice = {
        mixed_sublattice: tuple(
            component.strip().upper()
            for component in site_entries[mixed_sublattice].split(',')
            if component.strip()
        )
        for mixed_sublattice in mixed_sublattices
    }
    conditioners = tuple(
        (index, entry)
        for index, entry in enumerate(site_entries)
        if index not in mixed_sublattices
    )
    if len(mixed_sublattices) == 1:
        mixed_sublattice = mixed_sublattices[0]
        mixed_components = mixed_components_by_sublattice[mixed_sublattice]
        if len(mixed_components) == 2:
            term = BinaryExcessTerm(
                sublattice=mixed_sublattice,
                pair=mixed_components,
                order=order,
                conditioners=conditioners,
                interaction=interaction,
            )
            term.validate(context)
            return term
        if len(mixed_components) == 3:
            term = TernaryExcessTerm(
                sublattice=mixed_sublattice,
                triplet=mixed_components,
                order=order,
                conditioners=conditioners,
                interaction=interaction,
                is_l0_only=ternary_l0_only,
            )
            term.validate(context)
            return term
        raise ValueError(
            f"Unsupported restricted CEF TDB parameter site entry: {command}"
        )

    mixed_sublattice1, mixed_sublattice2 = mixed_sublattices
    mixed_components1 = mixed_components_by_sublattice[mixed_sublattice1]
    mixed_components2 = mixed_components_by_sublattice[mixed_sublattice2]
    if len(mixed_components1) == 2 and len(mixed_components2) == 2:
        term = TwoSublatticeBinaryExcessTerm(
            sublattice=(mixed_sublattice1, mixed_sublattice2),
            pair1=mixed_components1,
            pair2=mixed_components2,
            order=order,
            conditioners=conditioners,
            interaction=interaction,
        )
        term.validate(context)
        return term
    raise ValueError(
        f"Unsupported restricted CEF TDB parameter site entry: {command}"
    )


def get_cef_surface_from_tdb_string(
    tdb_string: str,
    phase_name: str,
    parameter_name: str,
    context: CEFContext,
    *,
    temperature_ref: float = 1000,
    correction_order: int | None = None,
    initial_parameter_scale: float = 0.1,
    interaction_factory: Callable[[str], TemperatureExpression] | None = None,
) -> CEFSurface:
    phase_name = phase_name.upper()
    parameter_name = parameter_name.upper()
    parameter_commands = []
    parameter_keys = []

    for command in _tdb_commands(tdb_string):
        parsed = _parse_tdb_parameter_command(command, parameter_name)
        if parsed is None:
            continue
        parameter_phase, site_array, order, _ = parsed
        if parameter_phase == phase_name:
            parameter_commands.append(command)
            parameter_keys.append((parameter_phase, site_array, order))

    orders_by_site = {}
    for parameter_phase, site_array, order in parameter_keys:
        orders_by_site.setdefault((parameter_phase, site_array), set()).add(order)

    terms = []
    for command in parameter_commands:
        parameter_phase, site_array, _, _ = _parse_tdb_parameter_command(
            command,
            parameter_name,
        )
        term = get_excess_term_from_tdb_string(
            command,
            context,
            parameter_name=parameter_name,
            temperature_ref=temperature_ref,
            correction_order=correction_order,
            initial_parameter_scale=initial_parameter_scale,
            ternary_l0_only=(orders_by_site[(parameter_phase, site_array)] == {0}),
            interaction_factory=interaction_factory,
        )
        if term is not None:
            terms.append(term)

    if not terms:
        raise ValueError(
            f"No {parameter_name} CEF terms were found for phase {phase_name!r}."
        )
    return CEFSurface(terms)


def get_magnetic_parameters_from_tdb_string(
    tdb_string: str,
    phase_name: str,
) -> tuple[float, float] | None:
    _, type_symbol = _get_cef_context_and_type_symbol(tdb_string, phase_name)
    if type_symbol is None:
        return None

    number = r'[-+0-9.Ee]+'
    pattern = re.compile(
        rf'TYPE(?:-DEF|_DEFINITION)\s+{re.escape(type_symbol)}\s+'
        rf'.*?\bMAGNETIC\s+({number})\s+({number})\s*$',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for command in _tdb_commands(tdb_string):
        match = pattern.match(command.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def get_cef_context_and_contributions_from_tdb_string(
    tdb_string: str,
    phase_name: str,
    *,
    temperature_ref: float = 1000,
    correction_order: int | None = None,
    zero_initialize_corrections: bool = False,
    interaction_factory: Callable[[str], TemperatureExpression] | None = None,
) -> tuple[CEFContext, Sequence[CEFContribution]]:
    context = get_cef_context_from_tdb_string(tdb_string, phase_name)
    initial_parameter_scale = 0.0 if zero_initialize_corrections else 0.1
    energy_surface = get_cef_surface_from_tdb_string(
        tdb_string,
        phase_name,
        "G",
        context,
        temperature_ref=temperature_ref,
        correction_order=correction_order,
        initial_parameter_scale=initial_parameter_scale,
        interaction_factory=interaction_factory,
    )
    contributions: list[CEFContribution] = [energy_surface]

    magnetic_parameters = get_magnetic_parameters_from_tdb_string(tdb_string, phase_name)
    if magnetic_parameters is not None:
        alpha, f = magnetic_parameters
        tc_surface = get_cef_surface_from_tdb_string(
            tdb_string,
            phase_name,
            "TC",
            context,
            temperature_ref=temperature_ref,
            correction_order=correction_order,
            initial_parameter_scale=initial_parameter_scale,
            interaction_factory=interaction_factory,
        )
        beta_surface = get_cef_surface_from_tdb_string(
            tdb_string,
            phase_name,
            "BM",
            context,
            temperature_ref=temperature_ref,
            correction_order=correction_order,
            initial_parameter_scale=initial_parameter_scale,
            interaction_factory=interaction_factory,
        )
        contributions.append(
            CEFMagneticContribution(
                beta_surface,
                tc_surface,
                alpha=alpha,
                f=f,
            )
        )
    return context, tuple(contributions)


class SymbolicCEFParser:
    """Construct CEF models whose TDB parameter expressions are symbolic modules."""

    def __init__(
        self,
        filename: str | Path,
        *,
        initial_parameter_scale: float = 0.1,
    ):
        self.path = Path(filename)
        self.tdb_string = self.path.read_text()
        self.initial_parameter_scale = float(initial_parameter_scale)
        self.phase_names = tuple(
            match.group(1).upper()
            for command in _tdb_commands(self.tdb_string)
            if (match := re.match(r"PHASE\s+([^\s]+)", command, flags=re.IGNORECASE))
        )


    def _create_expression(self, expression: str) -> ParameterizedExpression:
        return ParameterizedExpression(
            expression,
            initial_parameter_scale=self.initial_parameter_scale,
        )


    def get_model_for_phase(
        self,
        phase_name: str,
        config: "CEFConfig | None" = None,
        **kwargs,
    ) -> "CEF":
        from .cef_model import CEF

        phase_name = phase_name.upper()
        context, contributions = get_cef_context_and_contributions_from_tdb_string(
            self.tdb_string,
            phase_name,
            interaction_factory=self._create_expression,
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
        return CEF(
            components,
            context.sublattice_multiplicities,
            contributions=contributions,
            config=config,
            name=phase_name,
            **kwargs,
        )
