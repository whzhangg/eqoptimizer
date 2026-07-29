from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
from ase import io
from scipy.constants import Avogadro, electron_volt


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from eam_opt.calc import TorchEAMFSCalculator
from eam_opt.utilities import relax_structure
from eam_opt.models import CompPhase, SolutionPhase
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium
from eqopt.optimize import OptimizationConfig
from eqopt.models import EnsembleSystem



def relaxed_comp_phase(
    phase_name: str,
    structure_path: Path,
    calculator: TorchEAMFSCalculator,
    *,
    hydro: bool,
    elements: tuple[str, ...] | None = None,
) -> CompPhase:
    atoms = io.read(structure_path)
    relaxed = relax_structure(calculator, atoms, f_max=2.0e-3, hydro=hydro)
    symbols, counts = np.unique(relaxed.get_chemical_symbols(), return_counts=True)
    total = float(np.sum(counts))

    return CompPhase(
        phase_name,
        {
            symbol.upper(): float(count / total)
            for symbol, count in zip(symbols, counts, strict=True)
        },
        relaxed.get_potential_energy() / len(relaxed) * electron_volt * Avogadro,
        elements=elements,
    )


def build_models() -> tuple[SolutionPhase, CompPhase, CompPhase, CompPhase]:
    calculator = TorchEAMFSCalculator(str(THIS_DIR / "resources" / "CuAu_fitted.eam.fs"))
    structure_dir = THIS_DIR / "resources"
    elements = ("AU", "CU")

    fcc_entries = [
        relaxed_comp_phase(
            "FCC_AU",
            structure_dir / "fcc_Au4.cif",
            calculator,
            hydro=True,
            elements=elements,
        ),
        relaxed_comp_phase(
            "FCC_AU3CU",
            structure_dir / "fcc_Au12Cu4.cif",
            calculator,
            hydro=True,
            elements=elements,
        ),
        relaxed_comp_phase(
            "FCC_AUCU",
            structure_dir / "fcc_Cu8Au8.cif",
            calculator,
            hydro=True,
            elements=elements,
        ),
        relaxed_comp_phase(
            "FCC_AUCU3",
            structure_dir / "fcc_Cu12Au4.cif",
            calculator,
            hydro=True,
            elements=elements,
        ),
        relaxed_comp_phase(
            "FCC_CU",
            structure_dir / "fcc_Cu4.cif",
            calculator,
            hydro=True,
            elements=elements,
        ),
    ]
    fcc = SolutionPhase("FCC", fcc_entries)

    aucu = relaxed_comp_phase(
        "AUCU",
        structure_dir / "l10_Au2Cu2.cif",
        calculator,
        hydro=False,
    )
    aucu3 = relaxed_comp_phase(
        "AUCU3",
        structure_dir / "l12_AuCu3.cif",
        calculator,
        hydro=False,
    )
    au3cu = relaxed_comp_phase(
        "AU3CU",
        structure_dir / "l12_Au3Cu.cif",
        calculator,
        hydro=False,
    )
    return fcc, aucu, aucu3, au3cu


def get_tdb_str() -> str:
    models = build_models()
    return "\n\n".join([model.get_tdb_str() for model in models])


def get_observation(
    tdb_file: str,
    temp = Sequence[float]
) -> Sequence[PhaseEquilibrium]:
    """get PhaseEquilibrium"""
    handler = TDBHandler(tdb_file)
    all_data = []
    for t in temp:
        all_data += handler.build_equilibrium_data(t)
    return all_data


def optimize():
    REF = 'cuau_target.tdb'
    from eqopt.optimize import optimize_thermodynamic_parameters
    from eqopt.phase import PhaseID

    fcc, aucu, aucu3, au3cu = build_models()
    all_phases = {
        PhaseID(name='FCC', elements=['AU','CU']): fcc,
        PhaseID(name='AUCU', elements=['AU','CU']): aucu,
        PhaseID(name='AUCU3', elements=['AU','CU']): aucu3,
        PhaseID(name='AU3CU', elements=['AU','CU']): au3cu
    }
    system = EnsembleSystem(all_phases)
    eqilibrium = get_observation(REF, temp=[200, 400, 500, 550, 600, 650])
    
    # step 3. define configuration
    config = OptimizationConfig(
        epochs=1000,
        lr=100, 
        cosine_decay=True,
        min_lr_factor=0.2,
        regularization_weight=1e-10,
        use_huber_for_stable_phases=True,
    )
    
        # step 4. optimize
    optimized_system, equilibrium_states, optimization_state = optimize_thermodynamic_parameters(
        system,
        config,
        equilibria=eqilibrium,
        checkpoint_dir='results/checkpoint_tdb'
    )
    for phase_id in optimized_system.phase_ids:
        print(f'$ {phase_id}')
        print(optimized_system.get_model_by_phase_id(phase_id).get_tdb_str())

    
if __name__ == "__main__":
    optimize()
