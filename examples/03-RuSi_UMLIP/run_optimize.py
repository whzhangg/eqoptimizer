from collections.abc import Sequence

import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from eqopt.tdb_reference import TDBHandler
from eqopt.loss_function import PhaseEquilibrium
from eqopt.optimize import optimize_thermodynamic_parameters, OptimizationConfig
from eqopt.models import CEF, EnsembleSystem


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

REF = 'CPDDB_RuSi.tdb'
TO_OPT = 'initial.tdb'


if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases and create a system
    phase_ids = TDBHandler(TO_OPT).get_phase_ids()
    all_phases = {}
    for phid in phase_ids:
        all_phases[phid] = CEF.from_tdb_and_phasename(
            TO_OPT, phid.name, correction_order=2, temperature_ref=1000
        )
    system = EnsembleSystem(all_phases)

    # step 2. get data
    eqilibrium = get_observation(
        REF, 
        temp=[400, 700, 1000, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200]
    )

    # step 3. define configuration
    config = OptimizationConfig(
        epochs=500,
        lr=150, latent_mu_lr=150, cosine_decay=True, regularization_weight=1.0e-10,
        unstable_huber_beta=2.5,
    )

    # step 4. optimize
    optimized_system, equilibrium_states, optimization_state = optimize_thermodynamic_parameters(
        system,
        config,
        equilibria=eqilibrium,
        record_every=10
    )

    for phase_id in optimized_system.phase_ids:
        print(f'$ {phase_id}')
        print(optimized_system.get_model_by_phase_id(phase_id).get_tdb_str())
