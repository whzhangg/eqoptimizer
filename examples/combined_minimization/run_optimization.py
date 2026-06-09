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


REF = 'CPDDB_RuSi.tdb'
TO_OPT = 'initial.tdb'

if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases
    all_phases = []

    # RuSi
    TO_OPT = '../RuSi_UMLIP/initial.tdb'
    ref_db = Database(TO_OPT)
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=ref_db.elements,
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, temperature_ref=1800)
        ))
    # CuMg
    TO_OPT = '../CuMg_UMLIP/initial.tdb'
    ref_db = Database(TO_OPT)
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=ref_db.elements,
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, temperature_ref=1000)
        ))
    # AlZn
    TO_OPT = '../AlZn_UMLIP/initial.tdb'
    ref_db = Database(TO_OPT)
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=ref_db.elements,
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, temperature_ref=600)
        ))

    # step 2. define loss function
    loss = PhaseEquilibriumLoss(
        all_phases, regularization_weight=1e-13, regularize_difference=True)
    
    # step 3. get phase equilibria
    eqilibrium = []
    eqilibrium += get_observation(
        '../RuSi_UMLIP/CPDDB_RuSi.tdb', 
        all_phases, temp=[400, 700, 1000, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,2200])
    eqilibrium += get_observation(
        '../CuMg_UMLIP/CPDDB_CuMg.tdb', 
        all_phases, temp=[500, 700, 900, 1050, 1100])
    eqilibrium += get_observation(
        '../AlZn_UMLIP/CPDDB_AlZn.tdb', 
        all_phases, temp=[400, 500, 600, 640, 700, 800, 900])
    

    # step 4. optimize
    optimize_thermodynamic_parameters(
        loss, eqilibrium, epochs=400, lr=100.0, print_every=25
    )
    
    for phase in all_phases:
        print(f'$--{phase.phase_name}--')
        print(phase.model.get_tdb_str())
    
