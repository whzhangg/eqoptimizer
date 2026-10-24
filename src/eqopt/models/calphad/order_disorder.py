import math
from collections.abc import Sequence

import torch

from .cef_model import CEF, CEFConfig
from .cef_surface import CEFContribution


class SplitCEF(CEF):
    """Ordered CEF partitioned against a disordered reference CEF."""

    def __init__(self,
        components_on_sublattices: Sequence[Sequence[str]],
        sublattice_multiplicities: Sequence[float],
        contributions: Sequence[CEFContribution],
        disordered_model: CEF,
        config: CEFConfig | None = None,
        *,
        name: str | None = None,
        **kwargs,
    ):
        n_ordering_sublattices = (
            len(components_on_sublattices)
            - len(disordered_model.components_on_sublattices)
            + 1
        ) # number of ordering sublattices q = N_o - N_d + 1

        if n_ordering_sublattices < 2:
            raise ValueError(
                "SplitCEF requires at least two ordered sublattices corresponding "
                "to the first disordered sublattice."
            )

        disordered_components = set(disordered_model.components_on_sublattices[0])
        ordered_components = []
        for sublattice, components in enumerate(components_on_sublattices):
            components = set(components)
            if sublattice < n_ordering_sublattices:
                # the first q ordering sublattices
                if not components <= disordered_components:
                    raise ValueError(
                        f"Ordering sublattice {sublattice} contains components outside "
                        f"the disordered substitutional set: "
                        f"{sorted(components - disordered_components)}."
                    )
                components = disordered_components
            else:
                # remaining non-ordering sublattices
                disordered_sublattice = sublattice - n_ordering_sublattices + 1
                expected = set(
                    disordered_model.components_on_sublattices[disordered_sublattice]
                )
                if components != expected:
                    raise ValueError(
                        f"Ordered sublattice {sublattice} does not match disordered "
                        f"sublattice {disordered_sublattice}: expected "
                        f"{sorted(expected)}, got {sorted(components)}."
                    )
            ordered_components.append(tuple(sorted(components)))

        # check multiplicity
        ordering_multiplicity = sum(
            sublattice_multiplicities[:n_ordering_sublattices]
        )
        multiplicity_scale = (
            ordering_multiplicity
            / disordered_model.sublattice_multiplicities[0]
        ) # c
        for disordered_sublattice in range(
            1,
            len(disordered_model.sublattice_multiplicities),
        ):
            ordered_sublattice = n_ordering_sublattices + disordered_sublattice - 1
            actual = float(sublattice_multiplicities[ordered_sublattice])
            expected = (
                multiplicity_scale
                * disordered_model.sublattice_multiplicities[disordered_sublattice]
            )
            if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
                raise ValueError(
                    f"Ordered sublattice {ordered_sublattice} has multiplicity "
                    f"{actual:g}; expected {expected:g} from the disordered model."
                )


        super().__init__(
            ordered_components,
            sublattice_multiplicities,
            contributions,
            config,
            name=name,
            **kwargs,
        )
        self.disordered_model = disordered_model
        self.n_ordering_sublattices = n_ordering_sublattices
        self.multiplicity_scale = multiplicity_scale

        # build linear matrix mapping from
        # y_o1, y_o2, \cdots, y_oq, y_2, \cdots y_n 
        # to y_d, y_2, \cdots, y_n (disordered map)
        # to y_d, \cdots, y_d, y_2, \cdots, y_n (ordered_limit_map)
        disordered_map, ordered_limit_map = self._build_coordinate_maps()
        self.register_buffer(
            "disordered_from_ordered",
            disordered_map,
            persistent=False,
        )
        self.register_buffer(
            "ordered_disordered_limit_from_ordered",
            ordered_limit_map,
            persistent=False,
        )


    def _energy_from_internal_dof(
        self,
        y: torch.Tensor,
        temperature: float,
        normalize_by_amount: bool,
    ) -> torch.Tensor:
        y = torch.as_tensor(
            y,
            device=self.multi_for_each_y.device,
            dtype=self.multi_for_each_y.dtype,
        )
        disordered_y = y @ self.disordered_from_ordered.T
        ordered_disordered_y = y @ self.ordered_disordered_limit_from_ordered.T

        ordered_energy = super()._energy_from_internal_dof(
            y,
            temperature,
            normalize_by_amount=True,
        )
        disordered_energy = self.disordered_model._energy_from_internal_dof(
            disordered_y,
            temperature,
            normalize_by_amount=True,
        )
        ordered_disordered_energy = super()._energy_from_internal_dof(
            ordered_disordered_y,
            temperature,
            normalize_by_amount=True,
        )
        energy = ordered_energy + disordered_energy - ordered_disordered_energy

        if normalize_by_amount:
            return energy
        real_atom_amount = self._get_amount_of_elements_from_y(y).sum(dim=-1)
        return energy * real_atom_amount


    def _build_coordinate_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_ordered_y = len(self.y_names)
        n_disordered_y = len(self.disordered_model.y_names)
        device = self.multi_for_each_y.device
        dtype = self.multi_for_each_y.dtype
        disordered_map = torch.zeros(
            (n_disordered_y, n_ordered_y),
            device=device,
            dtype=dtype,
        )
        ordered_limit_map = torch.zeros(
            (n_ordered_y, n_ordered_y),
            device=device,
            dtype=dtype,
        )

        ordering_multiplicity = sum(
            self.sublattice_multiplicities[:self.n_ordering_sublattices]
        )
        averaged_indices = {}
        for component in self.disordered_model.components_on_sublattices[0]:
            indices = tuple(
                self.y_names_to_index[(component, sublattice)]
                for sublattice in range(self.n_ordering_sublattices)
            )
            averaged_indices[component] = indices
            for sublattice, index in enumerate(indices):
                weight = self.sublattice_multiplicities[sublattice] / ordering_multiplicity
                disordered_index = self.disordered_model.y_names_to_index[(component, 0)]
                disordered_map[disordered_index, index] = weight

        for ordered_index, (component, sublattice) in enumerate(self.y_names):
            if sublattice < self.n_ordering_sublattices:
                for source_sublattice, source_index in enumerate(
                    averaged_indices[component]
                ):
                    weight = (
                        self.sublattice_multiplicities[source_sublattice]
                        / ordering_multiplicity
                    )
                    ordered_limit_map[ordered_index, source_index] = weight
            else:
                disordered_sublattice = sublattice - self.n_ordering_sublattices + 1
                disordered_index = self.disordered_model.y_names_to_index[
                    (component, disordered_sublattice)
                ]
                disordered_map[disordered_index, ordered_index] = 1.0
                ordered_limit_map[ordered_index, ordered_index] = 1.0

        return disordered_map, ordered_limit_map
