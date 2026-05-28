import torch
import typing
import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from pycalphad import Database
from eqopt.tdb_reference import TDBHandler, EquilibriumCompositions
from eqopt.pycalphad_interface.pycal_models import PycalphadReferenceModel
from eqopt.models import RedlichKisterModel, CorrectedGibbsModel, PycalphadGibbsModel

REF = '../CPDDB_PbSn.tdb'
TO_OPT = 'initial.tdb'
def get_observation(temp = [350, 425, 475, 550, 700]) -> typing.List[EquilibriumCompositions]:
    handler = TDBHandler(REF)
    all_data = []
    for t in temp:
        eq_data = handler.build_equilibrium_data(t)
        all_data += eq_data
    return all_data


def optimize():
    from eqopt.optimize import optimize_thermodynamic_parameters
    ref_db = Database(TO_OPT)

    phases = {}
    for phase in ['FCC_A1', 'BCT_A5']:
        reference_model=PycalphadReferenceModel(ref_db, phase)
        correction_model=RedlichKisterModel(
            n_components=2,polynomial_order=1,interaction_order=0,elements=reference_model.elements,
            init_scale=100
        )
        phases[phase] = CorrectedGibbsModel(
            reference_model=reference_model,
            correction_model=correction_model
        )
    reference_model = PycalphadReferenceModel(ref_db, 'LIQUID')
    phases['LIQUID'] = PycalphadGibbsModel(reference_model)
    data = get_observation()
    #print(data)
    #raise
    losses = optimize_thermodynamic_parameters(
        phases, data, steps=100, lr=100.0, print_every=10, loss_threshold=1e-3, 
        n_samples=258,
        stable_weight=1.0,
        unstable_weight=1.0,
        regularization_weight=0.0)
    for name in ['FCC_A1', 'BCT_A5']:
        print(f'--{name}--')
        print(phases[name].correction_model.print_parameters())
    
    #import pickle
    #with open('result.pkl', 'wb') as f:
    #    pickle.dump(phases, f)

optimize()