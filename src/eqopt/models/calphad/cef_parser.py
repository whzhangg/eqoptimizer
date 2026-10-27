from collections.abc import Callable, Sequence
import dataclasses
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
    PiecewiseParameterizedExpression,
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


@dataclasses.dataclass(frozen=True)
class _ParsedPiecewiseExpression:
    segments: tuple[tuple[float | None, str], ...]
    upper_temperature: float | None



def get_cef_context_from_tdb_string(tdb_string: str, phase_name: str) -> CEFContext:
    context, _ = _get_cef_context_and_type_symbols(tdb_string, phase_name)
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
    interaction_factory: Callable[[_ParsedPiecewiseExpression], TemperatureExpression]
    | None = None,
) -> CEFExcessTerm | None:
    parsed = _parse_tdb_parameter_command(command, parameter_name)
    if parsed is None:
        return None

    phase_name, site_array, order, parsed_expression = parsed
    if context.phase_name is not None and phase_name != context.phase_name.upper():
        return None

    site_entries = site_array.split(':')
    if len(site_entries) != context.nsublattice:
        raise ValueError(
            f"Expected {context.nsublattice} sublattice entries, "
            f"got {len(site_entries)}: {command}"
        )

    if interaction_factory is not None:
        interaction = interaction_factory(parsed_expression)
    elif correction_order is not None:
        expression = _single_expression(parsed_expression)
        parameter_name = parameter_name.upper()
        interaction = TempPolynomialwCorrection.from_expression(
            expression,
            list(range(correction_order + 1)),
            temperature_ref=temperature_ref,
            physical_scale=_CORRECTION_PHYSICAL_SCALES[parameter_name],
            initial_parameter_scale=initial_parameter_scale,
        )
    else:
        expression = _single_expression(parsed_expression)
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
    interaction_factory: Callable[[_ParsedPiecewiseExpression], TemperatureExpression]
    | None = None,
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
    _, type_symbols = _get_cef_context_and_type_symbols(tdb_string, phase_name)
    if not type_symbols:
        return None

    number = r'[-+0-9.Ee]+'
    pattern = re.compile(
        rf'TYPE(?:-DEF|_DEF(?:INITION)?)\s+(\S+)\s+'
        rf'.*?\bMAGNETIC\s+({number})\s+({number})\s*$',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for command in _tdb_commands(tdb_string):
        match = pattern.match(command.strip())
        if match and match.group(1).upper() in type_symbols:
            return float(match.group(2)), float(match.group(3))
    return None


def get_order_disorder_relations_from_tdb_string(
    tdb_string: str,
) -> dict[str, str]:
    """
    Return ordered-to-disordered phase relationships from type 
    definitions.
    
    it return a dictionary ordered -> disordered reference
    """
    pattern = re.compile(
        r'TYPE(?:-DEF|_DEF(?:INITION)?)\s+\S+\s+GES\s+'
        r'(?:AMEND_PHASE_DESCRIPTION|A_P_D)\s+'
        r'([^,\s]+)\s+(?:DIS_PART|DISORDERED_PART)\s+([^,\s]+)',
        flags=re.IGNORECASE,
    )
    relations = {}
    for command in _tdb_commands(tdb_string):
        match = pattern.match(command.strip())
        if match is None:
            continue
        ordered_phase = match.group(1).upper()
        disordered_phase = match.group(2).upper()
        if ordered_phase in relations and relations[ordered_phase] != disordered_phase:
            raise ValueError(
                f"Ordered phase {ordered_phase!r} has multiple disordered references."
            )
        relations[ordered_phase] = disordered_phase
    return relations


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
            match.group(1).upper().split(":", 1)[0]
            for command in _tdb_commands(self.tdb_string)
            if (match := re.match(r"PHASE\s+([^\s]+)", command, flags=re.IGNORECASE))
        )
        self.order_disorder_relations = get_order_disorder_relations_from_tdb_string(
            self.tdb_string
        )
        self.disordered_reference_phase_names = frozenset(
            self.order_disorder_relations.values()
        )

        # first parse functions
        self._function_definitions = {}
        for command in _tdb_commands(self.tdb_string):
            parsed_function = _parse_tdb_function_command(command)
            if parsed_function is not None:
                name, expression = parsed_function
                self._function_definitions[name] = expression

        # recursively construct functions
        self.functions: dict[str, PiecewiseParameterizedExpression] = {}
        self._functions_being_constructed: set[str] = set()
        for name in self._function_definitions:
            self._get_function(name)

        self.models: dict[str, "CEF"] = {}
        self._phases_being_constructed: set[str] = set()


    def _get_function(self, name: str) -> PiecewiseParameterizedExpression:
        name = name.upper()
        if name in self.functions:
            return self.functions[name]
        if name in self._functions_being_constructed:
            raise ValueError(f"Cyclic TDB FUNCTION dependency involving {name!r}.")

        self._functions_being_constructed.add(name)
        try:
            expression = self._function_definitions[name]
            function = self._create_expression(expression)
            self.functions[name] = function
            return function
        finally:
            self._functions_being_constructed.remove(name)


    def _create_expression(
        self,
        expression: _ParsedPiecewiseExpression,
    ) -> PiecewiseParameterizedExpression:
        return PiecewiseParameterizedExpression(
            [
                (
                    lower_temperature,
                    ParameterizedExpression(
                        segment_expression,
                        initial_parameter_scale=self.initial_parameter_scale,
                        dependencies=self._get_expression_dependencies(segment_expression),
                    ),
                )
                for lower_temperature, segment_expression in expression.segments
            ],
            upper_temperature=expression.upper_temperature,
        )


    def _get_expression_dependencies(
        self,
        expression: str,
    ) -> dict[str, PiecewiseParameterizedExpression]:
        return {
            symbol_name: self._get_function(symbol_name.upper())
            for symbol_name in ParameterizedExpression.symbol_names(expression)
            if symbol_name.upper() in self._function_definitions
        }


    def get_models(
        self,
        config: "CEFConfig | None" = None,
        **kwargs,
    ) -> dict[str, "CEF"]:
        """Construct all phases once and return them by phase name."""
        for phase_name in self.phase_names:
            self.get_model_for_phase(phase_name, config=config, **kwargs)
        return dict(self.models)


    def get_model_for_phase(
        self,
        phase_name: str,
        config: "CEFConfig | None" = None,
        **kwargs,
    ) -> "CEF":
        from .cef_model import CEF
        from .order_disorder import SplitCEF

        phase_name = phase_name.upper()
        if phase_name in self.models:
            return self.models[phase_name]
        if phase_name not in self.phase_names:
            raise ValueError(f"Unknown phase {phase_name!r}.")
        if phase_name in self._phases_being_constructed:
            raise ValueError(
                f"Cyclic order-disorder phase relationship involving {phase_name!r}."
            )

        self._phases_being_constructed.add(phase_name)
        # load parameters and context
        try:
            context, contributions = get_cef_context_and_contributions_from_tdb_string(
                self.tdb_string,
                phase_name,
                interaction_factory=self._create_expression,
            )
            components = _components_from_context(context)
            disordered_phase = self.order_disorder_relations.get(phase_name)
            if disordered_phase is None:
                model = CEF(
                    components,
                    context.sublattice_multiplicities,
                    contributions=contributions,
                    config=config,
                    name=phase_name,
                    **kwargs,
                )
            else:
                disordered_model = self.get_model_for_phase(
                    disordered_phase,
                    config=config,
                    **kwargs,
                )
                model = SplitCEF(
                    components,
                    context.sublattice_multiplicities,
                    contributions=contributions,
                    disordered_model=disordered_model,
                    config=config,
                    name=phase_name,
                    **kwargs,
                )
            self.models[phase_name] = model
            return model
        finally:
            self._phases_being_constructed.remove(phase_name)


def get_cef_context_and_contributions_from_tdb_string(
    tdb_string: str,
    phase_name: str,
    *,
    temperature_ref: float = 1000,
    correction_order: int | None = None,
    zero_initialize_corrections: bool = False,
    interaction_factory: Callable[[_ParsedPiecewiseExpression], TemperatureExpression]
    | None = None,
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
    has_tc_surface = _has_tdb_parameter(tdb_string, phase_name, "TC")
    has_beta_surface = _has_tdb_parameter(tdb_string, phase_name, "BM")
    if magnetic_parameters is not None and has_tc_surface != has_beta_surface:
        raise ValueError(
            f"Magnetic phase {phase_name!r} must define both TC and BM surfaces."
        )
    if magnetic_parameters is not None and has_tc_surface:
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


def _parse_tdb_parameter_command(
    command: str,
    parameter_name: str,
) -> tuple[str, str, int, _ParsedPiecewiseExpression] | None:
    parameter_pattern = re.escape(parameter_name)
    if parameter_name.upper() == "G":
        parameter_pattern = r"(?:G|L)"
    elif parameter_name.upper() == "BM":
        parameter_pattern = r"BM(?:AG(?:N)?)?"
    prefix = (
        rf'PAR(?:A(?:METER)?)?\s+{parameter_pattern}\(\s*'
        r'([^,\s]+)\s*,\s*'
        r'(.+?)\s*'
        r'(?:;\s*(\d+))?\s*\)\s*'
    )
    match = re.match(prefix + r'(.+)$', command.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    site_entries = tuple(
        _normalize_tdb_sublattice_entry(entry)
        for entry in match.group(2).split(':')
    )
    return (
        match.group(1).strip().upper(),
        _site_entries_to_string(site_entries),
        int(match.group(3) or 0),
        _parse_piecewise_expression(match.group(4)),
    )


def _site_entries_to_string(site_entries: Sequence[Sequence[str]]) -> str:
    return ":".join(",".join(entry) for entry in site_entries)


def _parse_tdb_function_command(
    command: str,
) -> tuple[str, _ParsedPiecewiseExpression] | None:
    # return a tuple of function names and expression
    match = re.match(
        r'FUN(?:CTION)?\s+([^\s]+)\s+(.+)$',
        command.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).upper(), _parse_piecewise_expression(match.group(2))


def _parse_piecewise_expression(text: str) -> _ParsedPiecewiseExpression:
    """
    parser expression into segments. for each segment, we keep its 
    starting temperature. The last temperature is ignored so we only
    keep N_temperature = N_segment
    """
    text = text.replace('#', '')
    number = r'[-+0-9.Ee]+'
    parts = [part.strip() for part in text.split(';')]
    first_part = parts[0]

    # first part, eg, 298.5 a+bT
    if first_part.startswith(',,'):
        first_temperature = None
        first_expression = first_part[2:].strip()
    else:
        first_match = re.match(
            rf'({number})\s+(.+)$',
            first_part,
            flags=re.DOTALL,
        )
        if first_match is None:
            raise ValueError(f"Invalid TDB temperature expression: {text}")
        first_temperature = float(first_match.group(1))
        first_expression = first_match.group(2).strip()

    segments = [(first_temperature, first_expression)]
    for part in parts[1:-1]:
        segment_match = re.match(
            rf'({number})\s+Y\s+(.+)$',
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if segment_match is None:
            raise ValueError(f"Invalid TDB temperature interval: {part}")
        segments.append(
            (float(segment_match.group(1)), segment_match.group(2).strip())
        )

    # final temperature
    final_part = parts[-1]
    if final_part.startswith(',,'):
        final_match = re.match(r',\s*,\s*N(?:\s+.*)?$', final_part, re.IGNORECASE)
        upper_temperature = None
    else:
        final_match = re.match(
            rf'({number})\s+N(?:\s+.*)?$',
            final_part,
            flags=re.IGNORECASE,
        )
        upper_temperature = (
            None if final_match is None else float(final_match.group(1))
        )
    if final_match is None:
        raise ValueError(f"Invalid final TDB temperature interval: {final_part}")
    return _ParsedPiecewiseExpression(tuple(segments), upper_temperature)


def _single_expression(expression: _ParsedPiecewiseExpression) -> str:
    if len(expression.segments) != 1:
        raise ValueError(
            "Piecewise TDB expressions require SymbolicCEFParser."
        )
    return expression.segments[0][1]


def _has_tdb_parameter(
    tdb_string: str,
    phase_name: str,
    parameter_name: str,
) -> bool:
    phase_name = phase_name.upper()
    for command in _tdb_commands(tdb_string):
        parsed = _parse_tdb_parameter_command(command, parameter_name)
        if parsed is not None and parsed[0] == phase_name:
            return True
    return False


def _components_from_context(context: CEFContext) -> tuple[tuple[str, ...], ...]:
    components_by_sublattice: list[list[tuple[int, str]]] = [
        [] for _ in range(context.nsublattice)
    ]
    for (component, sublattice), index in context.y_names_to_index.items():
        components_by_sublattice[sublattice].append((index, component))
    return tuple(
        tuple(component for _, component in sorted(sublattice_components))
        for sublattice_components in components_by_sublattice
    )


def _tdb_commands(text: str) -> list[str]:
    """return TDB lines ending with ! after removing comments"""
    stripped = re.sub(r'\$.*$', '', text, flags=re.MULTILINE)
    return [
        command.strip()
        for command in stripped.split('!')
        if command.strip()
    ]


def _normalize_tdb_sublattice_entry(entry: str) -> tuple[str, ...]:
    """Match pycalphad's TDB normalization: uppercase and sort per sublattice."""
    return tuple(
        sorted(
            component.strip().rstrip("%").upper()
            for component in re.split(r'[,\s]+', entry)
            if component.strip()
        )
    )


def _get_cef_context_and_type_symbols(
    tdb_string: str,
    phase_name: str,
) -> tuple[CEFContext, tuple[str, ...]]:
    phase_name = phase_name.upper()
    multiplicity = None
    components = None
    type_symbols = ()

    for command in _tdb_commands(tdb_string):
        upper_command = command.upper()
        if upper_command.startswith('PHASE'):
            tokens = upper_command.split()
            command_phase_name = tokens[1].split(":", 1)[0]
            if len(tokens) >= 4 and command_phase_name == phase_name:
                type_symbols = tuple(tokens[2])
                n_sublattices = int(tokens[3])
                multiplicity = tuple(
                    float(value)
                    for value in tokens[4:4 + n_sublattices]
                )

        if upper_command.startswith(('CONSTITUENT', 'CONST ')):
            match = re.match(
                r'CONST(?:ITUENT)?\s+(\s*\S+\s*)\s*:(.*):\s*$',
                upper_command,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match and match.group(1).strip().upper().split(":", 1)[0] == phase_name:
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
    return context, type_symbols
