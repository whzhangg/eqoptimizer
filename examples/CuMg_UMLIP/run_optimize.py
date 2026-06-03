import torch
import typing
import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from pycalphad import Database
from eqopt.tdb_reference import TDBHandler, EquilibriumCompositions
from eqopt.models import CEF

REF = '../CPDDB_CuMg.tdb'
TO_OPT = 'initial.tdb'
def get_observation(
    temp = [500, 700, 900, 1050, 1100]
) -> typing.List[EquilibriumCompositions]:
    handler = TDBHandler(REF)
    all_data = []
    for t in temp:
        eq_data = handler.build_equilibrium_data(t)
        all_data += eq_data
    return all_data


def optimize():
    from eqopt.optimize import optimize_thermodynamic_parameters
    ref_db = Database(TO_OPT)
    candidate_phases = ref_db.phases.keys()

    phases = {}
    for phase in candidate_phases:
        phases[phase] = CEF.from_tdb_and_phasename(TO_OPT, phase)
        
    data = get_observation()
    
    losses = optimize_thermodynamic_parameters(
        phases, data, steps=100, lr=100.0, print_every=20, #loss_threshold=1e-4, 
        n_samples=32,
        stable_weight=1.0,
        unstable_weight=1.0,
        tau=0.005)
    
    for phase in candidate_phases:
        print(phases[phase].get_tdb_str())
    

optimize()
