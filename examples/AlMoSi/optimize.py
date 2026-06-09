from collections.abc import Sequence
from pycalphad import Database

import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium, PhaseEntry, PhaseEquilibriumLoss
from eqopt.optimize import optimize_thermodynamic_parameters
from eqopt.models import CEF


def get_observation(
    tdb_file: str,
    all_phases: Sequence[PhaseEntry],
    temp = Sequence[float]
) -> Sequence[PhaseEquilibrium]:
    """get PhaseEquilibrium"""
    handler = TDBHandler(tdb_file)
    all_data = []
    for t in temp:
        for eq in handler.build_equilibrium_data(t, nsamples=2):
            all_data.append(eq.get_phase_equilibrium_from_phase_entries(all_phases))
    return all_data

REF = 'CPDDB.tdb'
TO_OPT = 'CPDDB.tdb'

if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases
    ref_db = Database(TO_OPT)
    all_phases = []
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=set(['RU','SI']),
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, temperature_ref=1500)
        ))
    
    # step 2. define loss function
    loss = PhaseEquilibriumLoss(
        all_phases, regularization_weight=1e-13, regularize_difference=True)
    
    # step 3. get phase equilibria
    eqilibrium = get_observation(
        REF, all_phases, 
        temp=[1800]
        #temp=[400,  600,  800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
    )
    
    # step 4. optimize
    optimize_thermodynamic_parameters(
        loss, eqilibrium, epochs=0, lr=100.0, print_every=25
    )
    