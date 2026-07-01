from collections.abc import Sequence
from pycalphad import Database

import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium
from eqopt.optimize import optimize_thermodynamic_parameters, OptimizationConfig
from eqopt.models import CEF, EnsembleSystem
import torch

from eqopt.optimize import optimize_thermodynamic_parameters

def get_observation(
    tdb_file: str,
    temp = Sequence[float]
) -> Sequence[PhaseEquilibrium]:
    """get PhaseEquilibrium"""
    handler = TDBHandler(tdb_file)
    all_data = []
    for t in temp:
        all_data += handler.build_equilibrium_data(t, nsamples=12)
    return all_data


REF = 'CPDDB.tdb'
TO_OPT = 'initial.tdb'

def start_optimization():
    all_phases = {}
    phase_ids = TDBHandler(TO_OPT).get_phase_ids()
    for phid in phase_ids:
        all_phases[phid] = CEF.from_tdb_and_phasename(
            TO_OPT, phid.name, correction_order=1, temperature_ref=1500
        )
    system = EnsembleSystem(all_phases)
    print('Models are created')

    equilibria = get_observation('CPDDB.tdb', [500, 1000, 1500, 2000, 2500])
    print(f'Data are prepared, len={len(equilibria)}')
    
    config = OptimizationConfig(
        epochs=1000,
        lr=1000, cosine_decay=True, min_lr_factor=0.2, regularization_weight=1.0e-14,
        scale_energy_by_rt=False,
        use_huber_for_stable_phases=True,
        mu_init_lr=10000
    )
    optimized_system, equilibrium_states, optimization_state = optimize_thermodynamic_parameters(
        system=system,
        config=config,
        equilibria=equilibria,
        print_final_results=False,
        checkpoint_dir='new'
    )

    for phase_id in optimized_system.system.phase_ids:
        print(f'$ {phase_id}')
        print(optimized_system.system.get_model_by_phase_id(phase_id).get_tdb_str())


def restart_optimization():
    config = OptimizationConfig(
        epochs=1000,
        lr=1000, cosine_decay=True, min_lr_factor=0.2, regularization_weight=1.0e-14,
        scale_energy_by_rt=False,
        mu_init_lr=10000
    )
    state = optimize_thermodynamic_parameters(
        system='checkpoint/model_best.pt',
        config=config,
        equilibrium_states='checkpoint/equilibria_best.pt',
        checkpoint_dir='restart1',
        print_final_results=False
    )
    for phase_id in state.system.phase_ids:
        print(f'$ {phase_id}')
        print(state.system.get_model_by_phase_id(phase_id).get_tdb_str())


def restart_optimization2():
    config = OptimizationConfig(
        epochs=300,
        lr=500, cosine_decay=False, min_lr_factor=0.2, regularization_weight=1.0e-14,
        scale_energy_by_rt=False,
    )
    state = optimize_thermodynamic_parameters(
        system='restart1/model_best.pt',
        config=config,
        equilibrium_states='restart1/equilibria_best.pt',
        checkpoint_dir='restart2',
        print_final_results=False
    )
    for phase_id in state.system.phase_ids:
        print(f'$ {phase_id}')
        print(state.system.get_model_by_phase_id(phase_id).get_tdb_str())



if __name__ == "__main__":
    start_optimization()