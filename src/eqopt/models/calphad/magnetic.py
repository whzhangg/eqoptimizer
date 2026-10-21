import torch

from ...utilities import R
from .cef_surface import CEFContribution, CEFSurface
from .cef_terms import CEFContext


def magnetic_free_energy_Inden_Hillert_Jarl(
    alpha: float,
    f: float,
    beta: torch.Tensor,
    tc: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Return the IHJ magnetic energy per mole of formula units."""
    temperature = torch.as_tensor(temperature, device=beta.device, dtype=beta.dtype)
    beta_effective = torch.where(beta < 0.0, beta / alpha, beta)
    tc_effective = torch.where(tc < 0.0, tc / alpha, tc)

    a = 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / f - 1.0)
    q = tc_effective / temperature
    below_transition = q > 1.0

    # Only evaluate each expression in its physical domain. In particular, this
    # avoids inverse powers of q at Tc = 0 in the inactive low-temperature branch.
    q_below = torch.where(below_transition, q, torch.ones_like(q))
    q_above = torch.where(below_transition, torch.ones_like(q), q)
    g_below = 1.0 - (
        79.0 / (140.0 * f) * q_below
        + 474.0 / 497.0 * (1.0 / f - 1.0)
        * (
            q_below ** -3 / 6.0
            + q_below ** -9 / 135.0
            + q_below ** -15 / 600.0
        )
    ) / a
    g_above = -(
        q_above**5 / 10.0
        + q_above**15 / 315.0
        + q_above**25 / 1500.0
    ) / a
    g = torch.where(below_transition, g_below, g_above)
    return R * temperature * torch.log1p(beta_effective) * g


class CEFMagneticContribution(CEFContribution):
    """Inden-Hillert-Jarl magnetic energy from CEF beta and Tc surfaces."""

    def __init__(
        self,
        beta_surface: CEFSurface,
        tc_surface: CEFSurface,
        *,
        alpha: float,
        f: float,
    ):
        super().__init__()
        self.beta_surface = beta_surface
        self.tc_surface = tc_surface
        self.alpha = float(alpha)
        self.f = float(f)


    def validate(self, context: CEFContext) -> None:
        self.beta_surface.validate(context)
        self.tc_surface.validate(context)


    def forward(
        self,
        y: torch.Tensor,
        temperature: float | torch.Tensor,
        context: CEFContext,
    ) -> torch.Tensor:
        beta = self.beta_surface(y, temperature, context)
        tc = self.tc_surface(y, temperature, context)
        return magnetic_free_energy_Inden_Hillert_Jarl(
            self.alpha,
            self.f,
            beta,
            tc,
            temperature,
        )


    def to_tdb_strings(self, context: CEFContext) -> tuple[str, ...]:
        tc_strings = tuple(
            term.to_tdb_str(context, parameter_name="TC")
            for term in self.tc_surface.terms
        )
        beta_strings = tuple(
            term.to_tdb_str(context, parameter_name="BM")
            for term in self.beta_surface.terms
        )
        return tc_strings + beta_strings

