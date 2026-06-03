import torch
import typing
import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from pycalphad import Database
from eqopt.tdb_reference import TDBHandler, EquilibriumCompositions
from eqopt.models import CEF

REF = '../CPDDB_RuSi.tdb'
TO_OPT = 'initial.tdb'
def get_observation(
    temp = [400, 700, 1000, 1300, 1400, 1600, 1700, 1900, 2200]
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
        phases, data, steps=500, lr=100.0, print_every=20, #loss_threshold=1e-4, 
        n_samples=128,
        stable_weight=1.0,
        unstable_weight=1.0,
        tau=0.1)
    
    for phase in candidate_phases:
        print(phases[phase].get_tdb_str())
    

optimize()
