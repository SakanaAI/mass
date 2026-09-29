"""Optional core-dependency check: real RHI entry point with a fake model backend."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mass.io import ROOT, read_json


@unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("pydantic"), "requires requirements-core.txt")
class RHICoreTests(unittest.TestCase):
    def test_final_comparison_retains_winner_without_another_proposal(self):
        from harness_improvement import iterate_multi_agent_prompt as rhi
        calls = []
        class FakeBackend:
            def __init__(self, **kwargs):
                self.backend = kwargs["backend"]
            def verify_model(self):
                pass
            def call_json(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(payload={"winner": "B", "rationale": "Synthetic test verdict."},
                                       raw_text='{"winner":"B","rationale":"Synthetic test verdict."}', metrics={})

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prompt = ROOT / "tasks/initial/query51_ourTeam.txt"
            template = rhi._read_query_template(prompt, start_marker=rhi.DEFAULT_START_MARKER, end_marker=rhi.DEFAULT_END_MARKER)
            out = base / "search"
            (out / "versions").mkdir(parents=True)
            for v in [0, 1]:
                (out / f"versions/multi_agent_design_v{v}.txt").write_text(template.design_block)
                (base / f"v{v}/query51_ourTeam").mkdir(parents=True)
            (base / "reference/query51").mkdir(parents=True)
            argv = ["rhi", "--query-file", str(prompt), "--current-version", "1", "--out-dir", str(out),
                    "--current-repo", str(base / "v1/query51_ourTeam"), "--previous-repo", str(base / "v0/query51_ourTeam"),
                    "--include-best-design", "--judge-vs-baseline", "--baseline-repo-root", str(base / "reference"),
                    "--champion-repo-v0-root", str(base / "v0"), "--champion-repo-root-pattern", str(base / "v{version}"),
                    "--llm-backend", "local-vllm", "--require-local-vllm", "--model", "fixture-model", "--evaluate-only"]
            with patch.object(sys, "argv", argv), patch.object(rhi, "JsonLLMBackend", FakeBackend):
                self.assertEqual(rhi.main(), 0)
            retained = read_json(out / "retained_workflow.json")
            self.assertEqual(retained["version"], 1)
            self.assertTrue(retained["beats_reference"])
            self.assertEqual(len(calls), 3)  # baseline-v0, baseline-v1, champion arbitration
            self.assertFalse((out / "updated_queries/query51_ourTeam-v2.txt").exists())


if __name__ == "__main__":
    unittest.main()
