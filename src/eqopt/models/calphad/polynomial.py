from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Mapping, Sequence

import sympy
import torch

from ..shared import scalar_temperature, temperature_powers
from ...dtype import DEFAULT_DEVICE, DEFAULT_TYPE


class TemperatureExpression(torch.nn.Module, ABC):
    """Scalar temperature-dependent expression used by CALPHAD terms."""

    @abstractmethod
    def g(self, temperature) -> torch.Tensor:
        """Evaluate the expression at a scalar temperature."""


    def forward(self, temperature) -> torch.Tensor:
        return self.g(temperature)


    @abstractmethod
    def get_expression(self) -> str:
        """Return the current expression with trainable values substituted."""


class TempPolynomial(TemperatureExpression):
    """a temperature polynomial"""
    def __init__(self, 
        input_parameters: Sequence[float], 
        temperature_ref: float = 1000.0
    ):
        super().__init__()
        input_parameters = torch.as_tensor(
            input_parameters, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        tpower = temperature_powers(
            1.0, len(input_parameters)-1, temperature_ref
        )

        self.coeffs = torch.nn.Parameter(input_parameters / tpower)
        self.temperature_ref = temperature_ref


    def g(self, temperature) -> torch.Tensor:
        temperature = scalar_temperature(temperature)
        return temperature_powers(
            temperature, len(self.coeffs)-1, self.temperature_ref
        ) @ self.coeffs


    def get_expression(self) -> str:
        actual_coeff =  temperature_powers(
            1.0, len(self.coeffs)-1, self.temperature_ref
        ) * self.coeffs.detach()
        parts = []
        for i, c in enumerate(actual_coeff.detach().cpu().reshape(-1)):
            coefficient = float(c)
            if i == 0:
                parts.append(f'{coefficient:+.8e}')
            elif i == 1:
                parts.append(f'{coefficient:+.8e}*T')
            else:
                parts.append(f'{coefficient:+.8e}*T**{i}')
        return ' '.join(parts)


    @classmethod
    def from_expression(
        cls, expression: str, temperature_ref: float = 1000.0
    ) -> "TempPolynomial":
        """
        initial from an expression such as `-8.9013E+05 -5.4691E+01*T -9.5158E-02*T**2`
        """
        expression = expression.replace(' ', '')
        if not expression:
            raise ValueError("Empty polynomial expression.")
        if expression[0] not in '+-':
            expression = '+' + expression
        terms = []
        term_start = 0
        for index in range(1, len(expression)):
            if expression[index] in '+-' and expression[index - 1].upper() != 'E':
                terms.append(expression[term_start:index])
                term_start = index
        terms.append(expression[term_start:])
        coeffs: dict[int, float] = {}
        for term in terms:
            sign = -1.0 if term[0] == '-' else 1.0
            body = term[1:]
            if '*T**' in body:
                coefficient_text, order_text = body.split('*T**', 1)
                order = int(order_text)
            elif '*T' in body:
                coefficient_text = body.split('*T', 1)[0]
                order = 1
            elif body == 'T':
                coefficient_text = '1'
                order = 1
            else:
                coefficient_text = body
                order = 0
            coefficient = sign * float(coefficient_text)
            coeffs[order] = coeffs.get(order, 0.0) + coefficient

        max_order = max(coeffs, default=0)
        return cls(
            [coeffs.get(order, 0.0) for order in range(max_order + 1)],
            temperature_ref=temperature_ref,
        )


class TempPolynomialwCorrection(TemperatureExpression):
    """A frozen temperature polynomial plus trainable correction terms."""
    def __init__(self, 
        input_parameters: Sequence[float],
        correction_orders: Sequence[int], 
        *,
        temperature_ref: float = 1000.0,
        physical_scale: float = 1.0,
        initial_parameter_scale: float = 0.1,
    ):
        super().__init__()
        input_parameters = torch.as_tensor(
            input_parameters, device=DEFAULT_DEVICE, dtype=DEFAULT_TYPE)
        correction_orders = tuple(int(order) for order in correction_orders)
        if any(order < 0 for order in correction_orders):
            raise ValueError("Correction orders must be non-negative integers.")
        tpower = temperature_powers(
            1.0, len(input_parameters)-1, temperature_ref
        )
        self.register_buffer(
            'constants',
            input_parameters / tpower,
        )
        self.register_buffer(
            'orders',
            torch.as_tensor(
                correction_orders,
                device=DEFAULT_DEVICE,
                dtype=torch.int64,
            )
        )        
        self.coeffs = torch.nn.Parameter(
            torch.randn(
                len(correction_orders),
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            ) * initial_parameter_scale
        )
        self.temperature_ref = temperature_ref
        self.physical_scale = float(physical_scale)


    def g(self, temperature) -> torch.Tensor:
        temperature = scalar_temperature(temperature)
        constant_value = (
            temperature_powers(
                temperature, len(self.constants)-1, self.temperature_ref
            )
            @ self.constants
        )
        if self.coeffs.numel() == 0:
            return constant_value
        correction_powers = temperature_powers(
            temperature,
            self.orders,
            self.temperature_ref,
        )
        return constant_value + self.physical_scale * (correction_powers @ self.coeffs)


    def get_expression(self) -> str:
        max_order = max(
            int(len(self.constants) - 1),
            int(self.orders.max().item()) if self.orders.numel() else 0,
        )
        actual_coeff = torch.zeros(
            max_order + 1,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_TYPE,
        )
        actual_coeff[:len(self.constants)] += (
            temperature_powers(
                1.0,
                len(self.constants)-1,
                self.temperature_ref,
            )
            * self.constants.detach()
        )
        correction_actual_coeff = (
            temperature_powers(1.0, self.orders, self.temperature_ref)
            * self.physical_scale
            * self.coeffs.detach()
        )
        actual_coeff.index_add_(0, self.orders, correction_actual_coeff)
        parts = []
        for i, c in enumerate(actual_coeff.detach().cpu().reshape(-1)):
            coefficient = float(c)
            if i == 0:
                parts.append(f'{coefficient:+.8e}')
            elif i == 1:
                parts.append(f'{coefficient:+.8e}*T')
            else:
                parts.append(f'{coefficient:+.8e}*T**{i}')
        return ' '.join(parts)


    @classmethod
    def from_expression(
        cls,
        expression: str,
        correction_orders: Sequence[int],
        *,
        temperature_ref: float = 1000.0,
        physical_scale: float = 1.0,
        initial_parameter_scale: float = 0.1,
    ) -> "TempPolynomialwCorrection":
        """
        initial from an expression such as `-8.9013E+05 -5.4691E+01*T -9.5158E-02*T**2`
        """
        expression = expression.replace(' ', '')
        if not expression:
            raise ValueError("Empty polynomial expression.")
        if expression[0] not in '+-':
            expression = '+' + expression
        terms = []
        term_start = 0
        for index in range(1, len(expression)):
            if expression[index] in '+-' and expression[index - 1].upper() != 'E':
                terms.append(expression[term_start:index])
                term_start = index
        terms.append(expression[term_start:])
        coeffs: dict[int, float] = {}
        for term in terms:
            sign = -1.0 if term[0] == '-' else 1.0
            body = term[1:]
            if '*T**' in body:
                coefficient_text, order_text = body.split('*T**', 1)
                order = int(order_text)
            elif '*T' in body:
                coefficient_text = body.split('*T', 1)[0]
                order = 1
            elif body == 'T':
                coefficient_text = '1'
                order = 1
            else:
                coefficient_text = body
                order = 0
            coefficient = sign * float(coefficient_text)
            coeffs[order] = coeffs.get(order, 0.0) + coefficient

        max_order = max(coeffs, default=0)
        return cls(
            [coeffs.get(order, 0.0) for order in range(max_order + 1)],
            correction_orders=correction_orders,
            temperature_ref=temperature_ref,
            physical_scale=physical_scale,
            initial_parameter_scale=initial_parameter_scale,
        )


class ParameterizedExpression(TemperatureExpression):
    """Complete symbolic branch with local parameters and named dependencies.

    Exp, log and ln are supported along with polynomial expressions. Symbols
    other than ``T``, ``P``, and named dependencies are treated as local
    parameters. Pressure is evaluated at the standard pressure of one bar.
    """

    _temperature_symbol = sympy.Symbol("T")
    _pressure_symbol = sympy.Symbol("P")
    _runtime_symbols = {_temperature_symbol, _pressure_symbol}
    _standard_pressure_pa = 1.0e5
    _allowed_functions = {sympy.exp, sympy.log}
    _torch_modules = {
        "exp": torch.exp,
        "log": torch.log,
    }

    def __init__(
        self,
        expression: str,
        initial_parameter_scale: float = 0.1,
        *,
        dependencies: Mapping[str, "PiecewiseParameterizedExpression"] | None = None,
    ):
        super().__init__()
        self.expression = self._parse_expression(expression)
        dependencies = dependencies or {}
        self.dependencies = torch.nn.ModuleDict(dependencies)
        self.dependency_symbols = tuple(
            sorted(
                (
                    symbol
                    for symbol in self.expression.free_symbols
                    if str(symbol) in dependencies
                ),
                key=str,
            )
        )
        self.dependency_names = tuple(str(symbol) for symbol in self.dependency_symbols)
        self.parameter_symbols = tuple(
            sorted(
                (
                    symbol
                    for symbol in self.expression.free_symbols
                    if symbol not in self._runtime_symbols
                    and symbol not in self.dependency_symbols
                ),
                key=str,
            )
        )
        self.parameter_names = tuple(str(symbol) for symbol in self.parameter_symbols)
        self.theta = torch.nn.Parameter(
            torch.randn(
                len(self.parameter_symbols),
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_TYPE,
            ) * initial_parameter_scale
        )
        # determine if cache the result
        # if there is trainable parameter that determine the value 
        # of this expression, it's value at given temperature is
        # cached.
        self.cache_enabled = (
            self.theta.numel() == 0
            and all(
                dependency.is_parameter_independent
                for dependency in self.dependencies.values()
            )
        )
        self._value_cache: dict[
            tuple[float, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._function = None
        self._compile_function()


    @classmethod
    def from_expression(
        cls,
        expression: str,
        initial_parameter_scale: float = 0.1,
        *,
        dependencies: Mapping[str, "PiecewiseParameterizedExpression"] | None = None,
    ) -> "ParameterizedExpression":
        return cls(
            expression,
            initial_parameter_scale=initial_parameter_scale,
            dependencies=dependencies,
        )


    @classmethod
    def _parse_expression(cls, expression: str) -> sympy.Expr:
        expression = expression.strip()
        if not expression:
            raise ValueError("Empty parameterized expression.")
        parsed = sympy.sympify(
            expression,
            locals={
                "T": cls._temperature_symbol,
                "P": cls._pressure_symbol,
                "exp": sympy.exp,
                "EXP": sympy.exp,
                "ln": sympy.log,
                "LN": sympy.log,
                "log": sympy.log,
                "LOG": sympy.log,
            },
        )
        cls._validate_expression(parsed)
        return parsed


    @classmethod
    def _validate_expression(cls, expression: sympy.Expr) -> None:
        if isinstance(expression, (sympy.Symbol, sympy.Number)):
            return
        if expression.func in (sympy.Add, sympy.Mul):
            for argument in expression.args:
                cls._validate_expression(argument)
            return
        if expression.func is sympy.Pow:
            base, exponent = expression.args
            if not isinstance(exponent, sympy.Number):
                raise ValueError("ParameterizedExpression powers require numeric exponents.")
            cls._validate_expression(base)
            return
        if expression.func in cls._allowed_functions and len(expression.args) == 1:
            cls._validate_expression(expression.args[0])
            return
        raise ValueError(
            f"Unsupported operation in ParameterizedExpression: {expression.func}."
        )


    @classmethod
    def symbol_names(cls, expression: str) -> tuple[str, ...]:
        parsed = cls._parse_expression(expression)
        return tuple(
            sorted(
                str(symbol)
                for symbol in parsed.free_symbols
                if symbol not in cls._runtime_symbols
            )
        )


    def g(self, temperature) -> torch.Tensor:
        return self._evaluate(scalar_temperature(temperature), {})


    def _evaluate(
        self,
        temperature: torch.Tensor,
        cache: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        key = id(self)
        if key in cache:
            return cache[key]
        value_cache_key = None
        if self.cache_enabled:
            value_cache_key = (
                float(temperature.detach().cpu()),
                temperature.device,
                temperature.dtype,
            )
            if value_cache_key in self._value_cache:
                value = self._value_cache[value_cache_key]
                cache[key] = value
                return value
        if self._function is None:
            self._compile_function()
        dependency_values = [
            self.dependencies[name]._evaluate(temperature, cache)
            for name in self.dependency_names
        ]
        value = self._function(
            *self.theta.unbind(),
            *dependency_values,
            temperature,
            temperature.new_tensor(self._standard_pressure_pa),
        )
        if not isinstance(value, torch.Tensor):
            value = temperature.new_tensor(value)
        if value_cache_key is not None:
            value = value.detach()
            self._value_cache[value_cache_key] = value
        cache[key] = value
        return value


    def _compile_function(self) -> None:
        self._function = sympy.lambdify(
            (
                *self.parameter_symbols,
                *self.dependency_symbols,
                self._temperature_symbol,
                self._pressure_symbol,
            ),
            self.expression,
            modules=[self._torch_modules, "math"],
        )


    def __getstate__(self):
        state = super().__getstate__()
        state["_function"] = None
        state["_value_cache"] = {}
        return state


    def get_expression(self) -> str:
        symbol_substitutions = {
            symbol: sympy.Float(float(parameter.detach().cpu()), 16)
            for symbol, parameter in zip(self.parameter_symbols, self.theta)
        }
        expression = sympy.sstr(self.expression.subs(symbol_substitutions))
        return expression.replace("log(", "LN(").replace("exp(", "EXP(")


class PiecewiseParameterizedExpression(TemperatureExpression):
    """Select a complete parameterized expression by temperature interval."""

    def __init__(
        self,
        segments: Sequence[tuple[float, ParameterizedExpression]],
    ):
        super().__init__()
        self.lower_temperatures = tuple(
            0.0 if index == 0 else float(lower_temperature)
            for index, (lower_temperature, _) in enumerate(segments)
        )
        self.segments = torch.nn.ModuleList([segment for _, segment in segments])


    @property
    def is_parameter_independent(self) -> bool:
        return all(segment.cache_enabled for segment in self.segments)


    def g(self, temperature) -> torch.Tensor:
        return self._evaluate(scalar_temperature(temperature), {})


    def _evaluate(
        self,
        temperature: torch.Tensor,
        cache: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        key = id(self)
        if key in cache:
            return cache[key]

        segment_index = bisect_right(
            self.lower_temperatures,
            float(temperature.detach().cpu()),
        ) - 1
        segment = self.segments[segment_index]
        value = segment._evaluate(temperature, cache)
        cache[key] = value
        return value


    def get_expression(self) -> str:
        parts = [self.segments[0].get_expression()]
        for lower_temperature, segment in zip(
            self.lower_temperatures[1:],
            self.segments[1:],
        ):
            parts.append(f"{lower_temperature:g} Y {segment.get_expression()}")
        return "; ".join(parts)
