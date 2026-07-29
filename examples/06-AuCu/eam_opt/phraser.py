"""Parser for LAMMPS/DYNAMO EAM Finnis-Sinclair potential files.

The module name follows the existing project file name.  The parser supports
the ``eam/fs`` layout used by LAMMPS setfl-style Finnis-Sinclair potentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EAMFSPotential:
    """Tabulated EAM/FS potential data.

    Attributes
    ----------
    symbols
        Element symbols in file order.
    nrho, drho, nr, dr, cutoff
        Grid metadata from the potential file.
    embedding
        ``embedding[element, rho_index]`` in eV.
    density
        ``density[target_element, source_element, r_index]``.  This orientation
        matches the EAM/FS density sum: ``rho_i += density[type_i, type_j](r)``.
    rphi
        ``rphi[element_i, element_j, r_index]`` containing the file's ``r*phi``
        pair tables.  It is symmetric in the first two indices.
    """

    comments: tuple[str, str, str]
    symbols: tuple[str, ...]
    atomic_numbers: np.ndarray
    masses: np.ndarray
    lattice_constants: np.ndarray
    lattice_types: tuple[str, ...]
    nrho: int
    drho: float
    nr: int
    dr: float
    cutoff: float
    embedding: np.ndarray
    density: np.ndarray
    rphi: np.ndarray

    @property
    def rho_grid(self) -> np.ndarray:
        return np.arange(self.nrho, dtype=float) * self.drho

    @property
    def r_grid(self) -> np.ndarray:
        return np.arange(self.nr, dtype=float) * self.dr


def read_eam_fs(filename: str | Path) -> EAMFSPotential:
    """Read a LAMMPS/DYNAMO ``eam/fs`` potential file."""

    path = Path(filename)
    with path.open("r", encoding="utf-8") as handle:
        comments = tuple(handle.readline().rstrip("\n") for _ in range(3))
        tokens = handle.readline().split()
        if not tokens:
            raise ValueError(f"{path} does not contain an element-count line")

        n_elements = int(tokens[0])
        symbols = tuple(tokens[1 : 1 + n_elements])
        if len(symbols) != n_elements:
            raise ValueError(
                f"{path} declares {n_elements} elements but lists {len(symbols)}"
            )

        grid = handle.readline().split()
        if len(grid) < 5:
            raise ValueError(f"{path} does not contain a valid grid line")
        nrho = int(grid[0])
        drho = float(grid[1])
        nr = int(grid[2])
        dr = float(grid[3])
        cutoff = float(grid[4])

        atomic_numbers = np.empty(n_elements, dtype=int)
        masses = np.empty(n_elements, dtype=float)
        lattice_constants = np.empty(n_elements, dtype=float)
        lattice_types: list[str] = []
        embedding = np.empty((n_elements, nrho), dtype=float)
        density = np.empty((n_elements, n_elements, nr), dtype=float) # (target, source)

        for source in range(n_elements):
            header = handle.readline().split()
            if len(header) < 4:
                raise ValueError(f"{path} has an incomplete element header")

            atomic_numbers[source] = int(header[0])
            masses[source] = float(header[1])
            lattice_constants[source] = float(header[2])
            lattice_types.append(header[3])

            embedding[source] = _read_float_block(handle, nrho, path)
            for target in range(n_elements):
                density[target, source] = _read_float_block(handle, nr, path)

        rphi = np.empty((n_elements, n_elements, nr), dtype=float)
        for i in range(n_elements):
            for j in range(i + 1):
                values = _read_float_block(handle, nr, path)
                rphi[i, j] = values
                rphi[j, i] = values

        trailing = handle.read().split()
        if trailing:
            raise ValueError(f"{path} has {len(trailing)} unexpected trailing values")

    return EAMFSPotential(
        comments=comments,
        symbols=symbols,
        atomic_numbers=atomic_numbers,
        masses=masses,
        lattice_constants=lattice_constants,
        lattice_types=tuple(lattice_types),
        nrho=nrho,
        drho=drho,
        nr=nr,
        dr=dr,
        cutoff=cutoff,
        embedding=embedding,
        density=density,
        rphi=rphi,
    )


def _read_float_block(handle, count: int, path: Path) -> np.ndarray:
    values: list[float] = []
    while len(values) < count:
        line = handle.readline()
        if line == "":
            raise ValueError(f"{path} ended while reading a block of {count} floats")
        values.extend(float(token) for token in line.split())
    if len(values) != count:
        raise ValueError(f"{path} has too many values in a fixed-size table block")
    return np.asarray(values, dtype=float)
