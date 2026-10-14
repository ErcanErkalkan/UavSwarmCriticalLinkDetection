"""Lightweight tests of the reproducible experiment specification."""
import json
from pathlib import Path
import unittest
from acld_p0_6_common import build_specs, flatten_units, assigned_units

class Spec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {
            'CampaignSpec': Spec,
            'rician_kdb_to_nakagami_m': lambda k: 1.,
            'MAIN_METHODS': ('acld_v3', 'tarjan_predict_guard'),
            'ABLATION_METHODS': ('acld_v3',),
        }

    def test_a2_six_workers(self):
        units=flatten_units(build_specs(self.ns,'A2'),'A2')
        self.assertEqual(len(units),720)
        shards=[assigned_units(units,i,6) for i in range(1,7)]
        self.assertEqual([len(p) for p in shards], [120]*6)
        self.assertEqual(set(map(repr,sum(shards,[]))), set(map(repr,units)))

    def test_c1_six_workers(self):
        units=flatten_units(build_specs(self.ns,'C1'),'C1')
        self.assertEqual(len(units),100)
        self.assertEqual([len(assigned_units(units,i,6)) for i in range(1,7)], [17,17,17,17,16,16])

    def test_notebook(self):
        doc=json.loads((Path(__file__).resolve().parents[1]/'acld_full_experimental_campaign_P0_4_corrected.ipynb').read_text(encoding='utf-8-sig'))
        for i in (2,3,4):self.assertEqual(doc['cells'][i]['cell_type'],'code')

if __name__ == '__main__':
    unittest.main()
