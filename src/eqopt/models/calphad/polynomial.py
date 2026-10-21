from abc import ABC, abstractmethod
from typing import Sequence

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
    """Restricted symbolic expression with trainable symbols other than ``T``."""

    _runtime_symbol = sympy.Symbol("T")
    _allowed_functions = {sympy.exp, sympy.log}
    _torch_modules = {
        "exp": torch.exp,
        "log": torch.log,
    }

    def __init__(self, expression: str, initial_parameter_scale: float = 0.1):
        super().__init__()
        self.expression = self._parse_expression(expression)
        self.parameter_symbols = tuple(
            sorted(
                (
                    symbol
                    for symbol in self.expression.free_symbols
                    if symbol != self._runtime_symbol
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
        self._function = None
        self._compile_function()


    @classmethod
    def from_expression(
        cls,
        expression: str,
        initial_parameter_scale: float = 0.1,
    ) -> "ParameterizedExpression":
        return cls(expression, initial_parameter_scale=initial_parameter_scale)


    @classmethod
    def _parse_expression(cls, expression: str) -> sympy.Expr:
        expression = expression.strip()
        if not expression:
            raise ValueError("Empty parameterized expression.")
        parsed = sympy.sympify(
            expression,
            locals={
                "T": cls._runtime_symbol,
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


    def g(self, temperature) -> torch.Tensor:
        temperature = scalar_temperature(temperature)
        if self._function is None:
            self._compile_function()
        value = self._function(*self.theta.unbind(), temperature)
        if isinstance(value, torch.Tensor):
            return value
        return temperature.new_tensor(value)


    def _compile_function(self) -> None:
        self._function = sympy.lambdify(
            (*self.parameter_symbols, self._runtime_symbol),
            self.expression,
            modules=[self._torch_modules, "math"],
        )


    def __getstate__(self):
        state = super().__getstate__()
        state["_function"] = None
        return state


    def get_expression(self) -> str:
        substitutions = {
            symbol: sympy.Float(float(parameter.detach().cpu()), 16)
            for symbol, parameter in zip(self.parameter_symbols, self.theta)
        }
        expression = sympy.sstr(self.expression.subs(substitutions))
        return expression.replace("log(", "LN(").replace("exp(", "EXP(")

        
