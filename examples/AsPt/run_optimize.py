import torch
import typing
import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from pycalphad import Database
from eqopt.tdb_reference import TDBHandler, EquilibriumCompositions
from eqopt.pycalphad_interface.pycal_models import PycalphadReferenceModel
from eqopt.models import RedlichKisterModel, CorrectedGibbsModel, PycalphadGibbsModel, CompoundModel

REF = '../CPDDB_AsPt.tdb'
TO_OPT = 'initial.tdb'
def get_observation(
    temp = [400, 600, 800, 1000, 1200, 1600]
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

    phases = {}
    for phase in ['AS2PT']:
        reference_model=PycalphadReferenceModel(ref_db, phase)
        correction_model=CompoundModel(2, 1, elements=reference_model.elements)
        
        phases[phase] = CorrectedGibbsModel(
            reference_model=reference_model,
            correction_model=correction_model
        )
    for phase in ['LIQUID', 'RHOMB_A7', 'FCC_A1']:
        reference_model = PycalphadReferenceModel(ref_db, phase)
        phases[phase] = PycalphadGibbsModel(reference_model)
    
    data = get_observation()
    #print(data)
    #raise
    #phase_equilibrium_loss_parts(phases, data, debug=True)
    
    losses = optimize_thermodynamic_parameters(
        phases, data, steps=500, lr=100.0, print_every=20, #loss_threshold=1e-4, 
        n_samples=258,
        stable_weight=1.0,
        unstable_weight=1.0,
        regularization_weight=0.0,
        tau=0.01)
    for name in ['AS2PT']:
        print(f'--{name}--')
        print(phases[name].correction_model.print_parameters())
    
    #phase_equilibrium_loss_parts(phases, data, debug=True)
    
    import pickle
    with open('result.pkl', 'wb') as f:
        pickle.dump(phases, f)

optimize()