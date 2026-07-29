from ase import Atoms
from ase.optimize import BFGS, GoodOldQuasiNewton, FIRE, LBFGS # different possible optimizers
from ase.filters import FrechetCellFilter
import numpy as np

def get_composition_for_ase_atoms(atoms: Atoms) -> dict[str, float]:
    symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    total = float(np.sum(counts))
    return {
                symbol.upper(): float(count / total)
                for symbol, count in zip(symbols, counts, strict=True)
            }


def relax_structure(
    calculator, 
    init_atoms: Atoms, 
    f_max=5e-3, 
    step_max=500, 
    optimizer_class = GoodOldQuasiNewton,
    cell_filter = FrechetCellFilter,
    logfile=None,
    hydro=False,
) -> Atoms:
    """
    Parameters
    ----------
    init_atoms: ase.Atoms
        structure to relax
    f_max: float
        maximal value for forces
    step_max: int
        maximal number of steps
    logfile: str
        if other than none, details will be written to log file
        
    Reference:
     - https://wiki.fysik.dtu.dk/ase/ase/optimize.html
     - ams-tools-dev/amstools/relaxation/relaxation.py
        """
        
    energy_eps = 1e-3
        
    atoms = init_atoms.copy()
    atoms.pbc = True
    atoms.calc = calculator
    e_init = atoms.get_potential_energy()
    ucf = cell_filter(atoms, hydrostatic_strain=hydro)
    opt = optimizer_class(ucf, logfile=logfile)
    try:
        opt.run(fmax=f_max, steps=step_max)
    except Exception as e:
        print("OPTIMIZATION error: ", e)
        atoms = init_atoms
    e_final = atoms.get_potential_energy()
        
    if e_final - e_init > energy_eps:
        print(
            "! BAD OPTIMIZATION, energy increases from {} to {}".format(e_init, e_final)
        )
        atoms = init_atoms
    return atoms