from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .models.system_abc import ThermodynamicSystem
from .phase import PhaseEquilibrium, PhaseID


def phase_equilibria_from_dataframe(
    dataframe: pd.DataFrame,
    system: ThermodynamicSystem,
    *,
    zero_tolerance: float = 1.0e-12,
) -> list[PhaseEquilibrium]:
    """Construct phase equilibria from one-row-per-phase training data."""
    required_columns = {"eq_index", "temperature", "phase_name"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training data is missing required columns: {missing}")

    composition_columns = _composition_columns(system)
    csv_composition_columns = {
        column
        for column in dataframe.columns
        if isinstance(column, str) and column.startswith("x_")
    }
    if csv_composition_columns != set(composition_columns):
        expected = ", ".join(composition_columns)
        actual = ", ".join(sorted(csv_composition_columns))
        raise ValueError(
            f"Composition columns do not match the thermodynamic system. "
            f"Expected [{expected}], got [{actual}]."
        )

    data = dataframe.copy()
    numeric_indices = pd.to_numeric(data["eq_index"], errors="raise")
    indices_are_integers = np.equal(numeric_indices, np.floor(numeric_indices)).all()
    if numeric_indices.isna().any() or not indices_are_integers:
        raise ValueError("eq_index values must be integers.")
    data["eq_index"] = numeric_indices.astype(int)

    phase_ids_by_name = _phase_ids_by_name(system)
    elements = tuple(sorted(system.elements))
    equilibria = []
    for eq_index in sorted(data["eq_index"].unique()):
        rows = data[data["eq_index"] == eq_index]
        temperatures = pd.to_numeric(rows["temperature"], errors="raise").to_numpy(dtype=float)
        if not np.allclose(temperatures, temperatures[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Equilibrium {eq_index} contains inconsistent temperatures.")

        phases = []
        phase_compositions = []
        for _, row in rows.iterrows():
            phase_name = row["phase_name"]
            if pd.isna(phase_name) or phase_name not in phase_ids_by_name:
                raise ValueError(f"Unknown phase name {phase_name!r} in equilibrium {eq_index}.")
            phase_id = phase_ids_by_name[phase_name]

            values = row[composition_columns]
            if values.isna().all():
                composition = None
            elif values.isna().any():
                raise ValueError(
                    f"Composition for phase {phase_name!r} in equilibrium {eq_index} "
                    "must be either complete or entirely blank."
                )
            else:
                global_composition = {
                    element: float(row[f"x_{element}"])
                    for element in elements
                }
                unsupported = {
                    element: value
                    for element, value in global_composition.items()
                    if element not in phase_id.elements and abs(value) > zero_tolerance
                }
                if unsupported:
                    raise ValueError(
                        f"Phase {phase_name!r} does not support the nonzero components "
                        f"{unsupported}."
                    )
                composition = {
                    element: global_composition[element]
                    for element in phase_id.elements
                }

            phases.append(phase_id)
            phase_compositions.append(composition)

        equilibria.append(
            PhaseEquilibrium(
                phases=tuple(phases),
                phase_compositions=tuple(phase_compositions),
                temperature=float(temperatures[0]),
            )
        )
    return equilibria


def phase_equilibria_from_csv(
    filename: str | Path,
    system: ThermodynamicSystem,
    *,
    zero_tolerance: float = 1.0e-12,
) -> list[PhaseEquilibrium]:
    """Read phase-equilibrium training data from a CSV file."""
    dataframe = pd.read_csv(filename)
    return phase_equilibria_from_dataframe(
        dataframe,
        system,
        zero_tolerance=zero_tolerance,
    )


def phase_equilibria_to_dataframe(
    equilibria: Sequence[PhaseEquilibrium],
    *,
    elements: Collection[str] | None = None,
) -> pd.DataFrame:
    """Convert phase equilibria to one-row-per-phase training data."""
    if elements is None:
        elements = {
            element
            for equilibrium in equilibria
            for phase_id in equilibrium.phases
            for element in phase_id.elements
        }
    ordered_elements = tuple(sorted(elements))
    composition_columns = [f"x_{element}" for element in ordered_elements]
    records = []

    for eq_index, equilibrium in enumerate(equilibria):
        for phase_id, composition in zip(
            equilibrium.phases,
            equilibrium.phase_compositions,
            strict=True,
        ):
            record = {
                "eq_index": eq_index,
                "temperature": equilibrium.temperature,
                "phase_name": phase_id.name,
            }
            if composition is None:
                record.update({column: np.nan for column in composition_columns})
            else:
                record.update(
                    {
                        f"x_{element}": float(composition.get(element, 0.0))
                        for element in ordered_elements
                    }
                )
            records.append(record)

    columns = ["eq_index", "temperature", "phase_name", *composition_columns]
    return pd.DataFrame.from_records(records, columns=columns)


def phase_equilibria_to_csv(
    equilibria: Sequence[PhaseEquilibrium],
    filename: str | Path,
    *,
    elements: Collection[str] | None = None,
) -> None:
    """Write phase-equilibrium training data to a CSV file."""
    dataframe = phase_equilibria_to_dataframe(
        equilibria,
        elements=elements,
    )
    dataframe.to_csv(filename, index=False)


def _phase_ids_by_name(system: ThermodynamicSystem) -> Mapping[str, PhaseID]:
    phases_by_name: dict[str, list[PhaseID]] = {}
    for phase_id in system.phase_ids:
        phases_by_name.setdefault(phase_id.name, []).append(phase_id)

    ambiguous_names = [name for name, phase_ids in phases_by_name.items() if len(phase_ids) > 1]
    if ambiguous_names:
        names = ", ".join(sorted(ambiguous_names))
        raise ValueError(f"Phase names are not unique in the thermodynamic system: {names}")
    return {name: phase_ids[0] for name, phase_ids in phases_by_name.items()}


def _composition_columns(system: ThermodynamicSystem) -> list[str]:
    return [f"x_{element}" for element in sorted(system.elements)]
