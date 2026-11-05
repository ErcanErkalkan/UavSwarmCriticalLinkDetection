"""Validate published ACLD evidence metadata without claiming unavailable raw replay."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "research_hub/09_VALIDATED_EARLY_WARNING_20261105/S2_onset_verified_summary.json"
GATE = ROOT / "research_hub/00_GOVERNANCE/CURRENT_EVIDENCE_20261105.md"

class RecordedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s2 = json.loads(S2.read_text(encoding="utf-8"))

    def test_source_scope(self):
        self.assertTrue(GATE.is_file())
        self.assertEqual(self.s2["verified_unit_sha256"], 40)
        self.assertEqual(self.s2["unique_seed_clusters"], 10)
        self.assertEqual(self.s2["fixed_scenario_strata"], 4)
        self.assertIn("Post-hoc", self.s2["interpretation"])
        self.assertEqual(len(self.s2["source_archive_sha256"]), 64)

    def test_onset_confusion_accounting(self):
        counts = self.s2["counts_and_metrics"]
        for method, v in counts.items():
            self.assertEqual(v["total"], 159960, method)
            self.assertEqual(v["connected_current"] + v["already_disconnected_current"], 159960)
            self.assertEqual(v["TP"] + v["FN"], 46)
            self.assertEqual(v["FP"] + v["TN"], 3050)
            self.assertEqual(v["TP"] + v["FN"] + v["FP"] + v["TN"], 3096)
            self.assertAlmostEqual(v["FPR_at_risk"], v["FP"] / 3050)
        self.assertEqual((counts["acld_v3"]["TP"], counts["acld_v3"]["FP"]), (45, 117))
        self.assertEqual((counts["tarjan_predict_guard"]["TP"], counts["tarjan_predict_guard"]["FP"]), (46, 712))
        self.assertEqual((counts["components_only"]["TP"], counts["components_only"]["FP"]), (40, 4))

    def test_cluster_bootstrap(self):
        intervals = self.s2["bootstrap_5000_whole_seed_clusters"]
        d = intervals["FPR_difference_ACLD_minus_Tarjan"]
        self.assertLess(d["lower_95"], d["upper_95"])
        self.assertLess(d["upper_95"], 0)

if __name__ == "__main__":
    unittest.main()
