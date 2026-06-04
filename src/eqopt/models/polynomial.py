import torch
from typing import Sequence
from ..dtype import DEFAULT_DEVICE, DEFAULT_TYPE
from .shared import (
    scalar_temperature,
    temperature_powers,
)

class TempPolynomial(torch.nn.Module):
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


    def forward(self, temperature) -> torch.Tensor:
        return self.g(temperature)
    

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
            [coeffs.get(order, 0.0) for order in range(max_order + 1)], temperature_ref=temperature_ref
        )

        