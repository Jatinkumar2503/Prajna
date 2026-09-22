"""Tests for detect_suspicious_results, verify_results_current and the linter's allow audit."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detect_suspicious_results as dsr   # noqa: E402
import lint_hardcoded_results as lint      # noqa: E402
import verify_results_current as vrc       # noqa: E402


def doc_with(**arrays):
    return {"tables": {"exp": {k: {"per_seed": v} for k, v in arrays.items()}}}


class TestDetector(unittest.TestCase):
    A = [2.11, 2.24, 1.98, 2.31, 2.05]            # "Model A" per-seed values
    B_INDEPENDENT = [0.97, 1.05, 0.91, 1.02, 0.99]

    def test_flags_constant_ratio_like_res_b_equals_res_a_times_0_42(self):
        b = [x * 0.42 for x in self.A]
        flags = dsr.analyse(doc_with(model_a=self.A, model_b=b))
        self.assertTrue(any(f.startswith("C2") and ("0.42" in f or "2.381" in f) for f in flags), flags)

    def test_flags_constant_ratio_0_35_between_models(self):
        c = [x * 0.35 for x in self.B_INDEPENDENT]
        flags = dsr.analyse(doc_with(model_b=self.B_INDEPENDENT, model_c=c))
        self.assertTrue(any(f.startswith("C2") for f in flags), flags)

    def test_flags_zero_variance_unless_deterministic(self):
        flags = dsr.analyse(doc_with(prajna=[5.36] * 5))
        self.assertTrue(any(f.startswith("C1") for f in flags))
        det = {"tables": {"cusum": {"deterministic": True, "per_seed": [9.94] * 5}}}
        self.assertEqual(dsr.analyse(det), [])

    def test_independent_arrays_pass(self):
        self.assertEqual(dsr.analyse(doc_with(model_a=self.A, model_b=self.B_INDEPENDENT)), [])

    def test_flags_affine_relation(self):
        b = [1.7 * x + 0.3 for x in self.A]
        flags = dsr.analyse(doc_with(model_a=self.A, model_b=b))
        self.assertTrue(any(f.startswith(("C2", "C3")) for f in flags), flags)


class TestResultsCurrent(unittest.TestCase):
    def _git(self, repo, *args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                       env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"],
                            "HOME": str(repo)})

    def test_stale_results_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            self._git(repo, "init", "-q")
            (repo / "scripts").mkdir()
            (repo / "scripts" / "eval.py").write_text("x = 1\n")
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-q", "-m", "code")
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
            (repo / "results").mkdir()
            res = repo / "results" / "results.json"
            res.write_text(json.dumps({"metadata": {"git": {"git_hash": commit, "is_dirty": False}}}))
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-q", "-m", "results")
            self.assertEqual(vrc.check(str(res), ["scripts"], str(repo)), [])          # results commit alone is fine
            (repo / "scripts" / "eval.py").write_text("x = 2\n")
            self.assertTrue(any("uncommitted" in p for p in vrc.check(str(res), ["scripts"], str(repo))))
            self._git(repo, "add", "-A")
            self._git(repo, "commit", "-q", "-m", "change eval")
            self.assertTrue(any("changed since results" in p for p in vrc.check(str(res), ["scripts"], str(repo))))

    def test_missing_commit_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "r.json"
            res.write_text("{}")
            self.assertEqual(vrc.check(str(res), ["scripts"], d), ["results.json does not record a git commit"])


class TestAllowAudit(unittest.TestCase):
    def test_lists_and_caps_suppressions(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.py").write_text("acc_ref = 0.5  # provenance: allow (cited baseline)\nx = 1\n")
            self.assertEqual(len(lint.list_allows([d])), 1)
            self.assertEqual(lint.main([d, "--max-allows", "1"]), 0)
            self.assertEqual(lint.main([d, "--max-allows", "0"]), 1)


if __name__ == "__main__":
    unittest.main()
