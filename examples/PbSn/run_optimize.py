import torch
import typing
import sys
from pathlib import Path
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from pycalphad import Database
from eqopt.tdb_reference import TDBHandler, EquilibriumCompositions
from eqopt.models import CEF
from eqopt.utilities import R

REF = 'CPDDB_PbSn.tdb'
TO_OPT = 'initial.tdb'
def get_observation(temp = [350, 425, 475, 550, 700]) -> typing.List[EquilibriumCompositions]:
    handler = TDBHandler(REF)
    all_data = []
    for t in temp:
        eq_data = handler.build_equilibrium_data(t)
        all_data += eq_data
    return all_data


def optimize():
    from eqopt.optimize import optimize_thermodynamic_parameters, freeze_model
    CEF.cef_penaltyweight = 1e8
    phases = {}
    for phase in ['FCC_A1', 'BCT_A5', 'LIQUID']:
        phases[phase] = CEF.from_tdb_and_phasename(TO_OPT, phase)
        
    for k, v in phases.items():
        print(k)
        print(v.get_tdb_str())
    freeze_model(phases['LIQUID'])
    data = get_observation()
    
    losses = optimize_thermodynamic_parameters(
        phases, data, steps=200, lr=100, print_every=10, loss_threshold=1e-5,
        tau=0.1,
        n_samples=256, regularization_weight=0.0)
    
    for name in ['FCC_A1', 'BCT_A5']:
        print(phases[name].get_tdb_str())

optimize()
