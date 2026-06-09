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

REF = 'CPDDB_WRe.tdb'
TO_OPT = 'initial.tdb'
#TO_OPT = 'optimized.tdb'

if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases
    ref_db = Database(TO_OPT)
    all_phases = []
    for phase in ref_db.phases.keys():
        all_phases.append(PhaseEntry(
            phase_name=phase,
            elements=set(['W','RE']),
            model=CEF.from_tdb_and_phasename(TO_OPT, phase, temperature_ref=2000, correction_order=1)
        ))

    #chi = PhaseEntry(
    #        phase_name=phase,
    #        elements=set(['W','RE']),
    #        model=CEF.from_tdb_and_phasename(TO_OPT, 'CHI', temperature_ref=3000)
    #)
    #print(chi.model.gibbs_energy_per_molar_atom({'RE':1-0.193,'W':0.193}, temperature=400))
    #raise

    # step 2. define loss function
    loss = PhaseEquilibriumLoss(
        all_phases, regularization_weight=1e-16, regularize_difference=True)
    
    # step 3. get phase equilibria
    eqilibrium = get_observation(
        REF, all_phases, temp=[400, 700, 1000, 1300, 1600, 1900, 2200, 2500, 2800, 3100, 3200, 3400, 3600])
    
    # step 4. optimize
    optimize_thermodynamic_parameters(
        #loss, eqilibrium, epochs=800, lr=1000.0, print_every=10
        loss, eqilibrium, epochs=800, lr=1000.0, print_every=10, cosine_decay=True, min_lr_factor=0.1
    )
    
    for phase in all_phases:
        print(f'$--{phase.phase_name}--')
        print(phase.model.get_tdb_str())
    
