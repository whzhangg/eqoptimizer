from collections.abc import Sequence
from pycalphad import Database

import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium
from eqopt.optimize import optimize_thermodynamic_parameters, OptimizationConfig
from eqopt.models import CEF, EnsembleSystem
from eqopt.phase import PhaseID


def get_observation(
    tdb_file: str,
    temp = Sequence[float]
) -> Sequence[PhaseEquilibrium]:
    """get PhaseEquilibrium"""
    handler = TDBHandler(tdb_file)
    all_data = []
    for t in temp:
        all_data += handler.build_equilibrium_data(t, nsamples=8)
    return all_data

TO_OPT = 'initial.tdb'
REF = 'CPDDB.tdb'

if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases and create a system
    all_phases = {}
    ref_db = Database(TO_OPT)
    phase_ids = TDBHandler(TO_OPT).get_phase_ids()
    for phid in phase_ids:
        all_phases[phid] = CEF.from_tdb_and_phasename(
            TO_OPT, phid.name, correction_order=1, temperature_ref=2000
        )
    system = EnsembleSystem(all_phases)

    # step 2. get data
    eqilibrium = get_observation(REF, temp=[1000, 1500, 2000, 2500, 3000])

    # step 3. define configuration
    config = OptimizationConfig(
        epochs=1000,
        lr=500, cosine_decay=True, regularization_weight=1.0e-14
    )

    # step 4. optimize
    state = optimize_thermodynamic_parameters(
        system,
        eqilibrium, 
        config=config
    )

    for phase_id in state.system.phase_ids:
        print(f'$ {phase_id}')
        print(state.system.get_model_by_phase_id(phase_id).get_tdb_str())
