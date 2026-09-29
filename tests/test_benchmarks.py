import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mass.io import ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"benchmarks/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BenchmarkTests(unittest.TestCase):
    def test_sab_evaluator_paths_survive_its_changed_working_directory(self):
        spec = importlib.util.spec_from_file_location("collector", ROOT / "benchmarks/scripts/bench_sab_collect_preds.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with patch.object(mod.subprocess, "run") as run:
            log = mod.evaluate(Path("relative_predictions"), "fixture", 1)
        args = run.call_args.args[0]
        self.assertTrue(Path(args[args.index("--pred_program_path") + 1]).is_absolute())
        self.assertTrue(log.is_absolute())

    def test_sab_uses_official_scores_and_requires_complete_trials(self):
        mod = load("summarize")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scores.jsonl"
            p.write_text('\n'.join(json.dumps({"success_rate": i % 2}) for i in range(102)))
            self.assertEqual(mod.sab_trial(p), .5)
            p.write_text('{"success_rate": 1}\n')
            with self.assertRaises(ValueError):
                mod.sab_trial(p)

    def test_mlr_missing_papers_zero_missing_judges_error(self):
        mod = load("summarize_mlr")
        reviews = {"a": {"job": "mlr_L0_t1", "task": "a", "reviews": {
            j: {"Overall": 0, "missing_paper": True} for j in ["gpt-5.5", "claude-opus-4-8"]}}}
        self.assertEqual(mod.summarize(reviews, {"a"})["L0"]["mean"], 0)
        reviews["a"]["reviews"].pop("gpt-5.5")
        with self.assertRaises(ValueError):
            mod.summarize(reviews, {"a"})


if __name__ == "__main__":
    unittest.main()
