from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from .cef_terms import CEFContext, CEFExcessTerm


class CEFContribution(nn.Module, ABC):
    """Additive contribution to a CEF formula-unit energy."""

    @abstractmethod
    def validate(self, context: CEFContext) -> None:
        """Validate the contribution against the phase coordinate system."""


    @abstractmethod
    def forward(
        self,
        y: torch.Tensor,
        temperature: float | torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        """Evaluate the contribution at the supplied internal coordinates."""


    @abstractmethod
    def to_tdb_strings(self, context: CEFContext) -> tuple[str, ...]:
        """Return the TDB parameter commands represented by this contribution."""


class CEFSurface(CEFContribution):
    """Scalar function of CEF internal coordinates assembled from CEF terms."""

    def __init__(self, terms: Sequence[CEFExcessTerm]):
        super().__init__()
        self.terms = tuple(terms)

        for term in self.terms:
            if not isinstance(term, CEFExcessTerm):
                raise TypeError(f"Unsupported CEF surface term {type(term).__name__}.")

        # CEF terms are lightweight rules rather than nn.Modules. Register their
        # temperature polynomials here so their parameters belong to this surface.
        self.interactions = nn.ModuleList([term.interaction for term in self.terms])


    def validate(self, context: CEFContext) -> None:
        for term in self.terms:
            term.validate(context)


    def forward(
        self,
        y: torch.Tensor,
        temperature: float | torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        value = torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        for term in self.terms:
            value = value + term.get_contribution(y, temperature, context)
        return value


    def to_tdb_strings(self, context: CEFContext) -> tuple[str, ...]:
        return tuple(term.to_tdb_str(context) for term in self.terms)
