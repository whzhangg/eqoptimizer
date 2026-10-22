import sys
sys.path.append('/Users/wenhao/work/projects/2026-optimize PD/src')
from eqopt.tdb_reference import TDBHandler
from eqopt.training_data import phase_equilibria_from_csv, phase_equilibria_to_csv


REF = 'CrFe-reference.tdb'


def write_training_data(
    temp=[250, 500, 750, 900, 1000, 1100, 1250, 1350, 1500, 1600, 1800, 1900, 2000]
):
    handler = TDBHandler(REF)
    all_data = []
    for t in temp:
        print(t)
        all_data += handler.build_equilibrium_data(t)
    phase_equilibria_to_csv(all_data, 'training_equilibria.csv')


if __name__ == "__main__":
    write_training_data()