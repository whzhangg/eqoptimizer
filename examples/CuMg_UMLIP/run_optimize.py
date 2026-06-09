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
        for eq in handler.build_equilibrium_data(t):
            all_data.append(eq.get_phase_equilibrium_from_phase_entries(all_phases))
    return all_data

REF = 'CPDDB_CuMg.tdb'
TO_OPT = 'initial.tdb'

if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases
    ref_db = Database(TO_OPT)
    all_phases = []
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=ref_db.elements,
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, correction_order=1)
        ))

    # step 2. define loss function
    loss = PhaseEquilibriumLoss(all_phases, regularization_weight=1e-11)
    
    # step 3. get phase equilibria
    eqilibrium = get_observation(REF, all_phases, temp=[500, 700, 900, 1050, 1100])
    
    # step 4. optimize
    optimize_thermodynamic_parameters(
        loss, eqilibrium, epochs=200, lr=100.0, print_every=25
    )
    
    for phase in all_phases:
        print(f'$--{phase.phase_name}--')
        print(phase.model.get_tdb_str())
    
