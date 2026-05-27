from tdb_ref import TDBHandler

TDB_ALZN = '../../examples/Al-Zn_CPDDB.tdb'

def test_Al_Zn():
    handler = TDBHandler(TDB_ALZN)

    assert len(handler.build_equilibrium_data(temperature=600)) == 2
    assert len(handler.build_equilibrium_data(temperature=400)) == 1
    assert len(handler.build_equilibrium_data(temperature=800)) == 1
    