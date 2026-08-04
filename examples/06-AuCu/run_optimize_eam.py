from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Mapping, Sequence

from ase import io


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from potentials.eam.model import FineTunedEAM, FineTunedEAMConfig
from eam_opt.utilities import relax_structure
from eam_opt.models import CompPhase, SolutionPhase, EAMCompPhase
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium
from eqopt.optimize import OptimizationConfig
from eqopt.models import EnsembleSystem


def build_models(eam_model) -> tuple[SolutionPhase, CompPhase, CompPhase, CompPhase]:
    structure_dir = THIS_DIR / "resources"
    fcc_entries = [
        EAMCompPhase('FCC_AU', io.read(structure_dir/'fcc_Au4.cif'), eam_model, hydro=True),
        EAMCompPhase('FCC_AU3CU', io.read(structure_dir/'fcc_Au12Cu4.cif'), eam_model, hydro=True),
        EAMCompPhase('FCC_AUCU', io.read(structure_dir/'fcc_Cu8Au8.cif'), eam_model, hydro=True),
        EAMCompPhase('FCC_AUCU3', io.read(structure_dir/'fcc_Cu12Au4.cif'), eam_model, hydro=True),
        EAMCompPhase('FCC_CU', io.read(structure_dir/'fcc_Cu4.cif'), eam_model, hydro=True)
    ]
    fcc = SolutionPhase("FCC", fcc_entries)

    aucu = EAMCompPhase('AUCU', io.read(structure_dir/'l10_Au2Cu2.cif'), eam_model)
    aucu3 = EAMCompPhase('AUCU3', io.read(structure_dir/'l12_AuCu3.cif'), eam_model)
    au3cu = EAMCompPhase('AU3CU', io.read(structure_dir/'l12_Au3Cu.cif'), eam_model)
    return fcc, aucu, aucu3, au3cu


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
    import torch
    from eqopt.optimize import optimize_thermodynamic_parameters
    from eqopt.phase import PhaseID

    # define the model
    eam_config = FineTunedEAMConfig()
    eam_config.n_density_basis = 6
    eam_config.n_pair_basis = 6

    eam = FineTunedEAM(
        THIS_DIR/'resources'/'CuAu_fitted.eam.fs', config=eam_config
    )

    REF = THIS_DIR/ 'cuau_target.tdb'
    
    fcc, aucu, aucu3, au3cu = build_models(eam)
    all_phases = {
        PhaseID(name='FCC', elements=['AU','CU']): fcc,
        PhaseID(name='AUCU', elements=['AU','CU']): aucu,
        PhaseID(name='AUCU3', elements=['AU','CU']): aucu3,
        PhaseID(name='AU3CU', elements=['AU','CU']): au3cu
    }
    system = EnsembleSystem(all_phases)
    eqilibrium = get_observation(REF, temp=[300, 400, 500, 550, 600, 650])
    
    # step 3. define configuration
    config = OptimizationConfig(
        epochs=1000,
        lr=5.0e-3, 
        cosine_decay=True,
        min_lr_factor=0.2,
        regularization_weight=1e-3,
        use_huber_for_stable_phases=True,
    )
    
        # step 4. optimize
    optimized_system, equilibrium_states, optimization_state = optimize_thermodynamic_parameters(
        system,
        config,
        equilibria=eqilibrium,
        checkpoint_dir='checkpoint_tmp'
    )
    for phase_id in optimized_system.phase_ids:
        print(f'$ {phase_id}')
        print(optimized_system.get_model_by_phase_id(phase_id).get_tdb_str())

    #torch.save(eam, 'results/optimized_eam.pt')
    
if __name__ == "__main__":
    optimize()
