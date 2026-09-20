"""
Unit test for CI: Verifies that every number in generated tables originates
from results/latest/results.json.
"""

import unittest
from pathlib import Path
from scripts.verify_table_provenance import verify_provenance_file
from scripts.seal_results import compute_canonical_hash

class TestTableProvenanceCI(unittest.TestCase):
    def setUp(self):
        self.workspace_root = Path(__file__).resolve().parent.parent
        self.provenance_json = self.workspace_root / "results" / "latest" / "results.json"
        self.provenance_md = self.workspace_root / "docs" / "provenance_tables.md"
        self.readme_md = self.workspace_root / "README.md"

    def test_01_provenance_json_exists_and_has_required_metadata(self):
        self.assertTrue(self.provenance_json.exists(), "results/latest/results.json must exist")
        import json
        with open(self.provenance_json, "r") as f:
            data = json.load(f)
        self.assertIn("metadata", data)
        self.assertIn("tables", data)
        meta = data["metadata"]
        self.assertIn("git", meta)
        self.assertIn("git_hash", meta["git"])
        self.assertIn("hardware", meta)
        self.assertIn("seeds", meta)
        self.assertIn("exact_command", meta)
        self.assertIn("dataset_and_source_hashes", meta)

    def test_02_provenance_tables_md_matches_json(self):
        self.assertTrue(self.provenance_md.exists(), "docs/provenance_tables.md must exist")
        passed = verify_provenance_file(self.provenance_md, self.provenance_json)
        self.assertTrue(passed, "Every table in docs/provenance_tables.md must match results.json byte-for-byte")

    def test_03_readme_tables_match_json_if_present(self):
        with open(self.readme_md, "r", encoding="utf-8") as f:
            content = f.read()
        if "<!-- PROVENANCE_TABLE_START:" in content:
            passed = verify_provenance_file(self.readme_md, self.provenance_json)
            self.assertTrue(passed, "Every table in README.md must match results.json byte-for-byte")

    def test_04_results_cryptographically_sealed(self):
        import json
        with open(self.provenance_json, "r") as f:
            data = json.load(f)
        seal = data.get("metadata", {}).get("seal", {})
        self.assertIn("canonical_sha256", seal, "results.json must contain cryptographic seal")
        computed = compute_canonical_hash(data)
        self.assertEqual(computed, seal["canonical_sha256"], "Cryptographic seal hash must match canonical JSON hash")

if __name__ == "__main__":
    unittest.main()
