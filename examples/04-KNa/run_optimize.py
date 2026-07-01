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
       
    for eq in all_data:
        i_liq = 0
        has_liquid = False
        for i, id in enumerate(eq.phases):
            if 'LIQUID' in id.name:
                i_liq = i
                has_liquid = True
                break
        
        if has_liquid:
            eq.phase_compositions[i_liq-1] = None
    
    return all_data


REF = 'CPDDB.tdb'
TO_OPT = 'initial.tdb'


if __name__ == "__main__":
    from eqopt.optimize import optimize_thermodynamic_parameters

    # step 1. get all phases and create a system
    phase_ids = TDBHandler(TO_OPT).get_phase_ids()
    all_phases = {}
    for phid in phase_ids:
        all_phases[phid] = CEF.from_tdb_and_phasename(
            TO_OPT, phid.name, correction_order=1, temperature_ref=200
        )
    system = EnsembleSystem(all_phases)

    # step 2. get data
    eqilibrium = get_observation(
        REF, temp=[50, 100, 150, 200, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340])

    # step 3. define configuration
    config = OptimizationConfig(
        epochs=3000,
        lr=50,
        latent_mu_lr=50, 
        cosine_decay=True,
        scale_energy_by_rt=False,
        use_huber_for_stable_phases=True, 
        regularization_weight=1e-8,
        mu_convergence_tol=5,
        unstable_huber_beta=2.5,
    )

    # step 4. optimize
    optimized_system, equilibrium_states, optimization_state = optimize_thermodynamic_parameters(
        system,
        config,
        equilibria=eqilibrium,
    )

    for phase_id in optimized_system.phase_ids:
        print(f'$ {phase_id}')
        print(optimized_system.get_model_by_phase_id(phase_id).get_tdb_str())
    
    
