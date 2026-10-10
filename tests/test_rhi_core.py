"""Optional core-dependency check: real RHI entry point with a fake model backend."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mass.io import ROOT, read_json


@unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("pydantic"), "requires requirements-core.txt")
class RHICoreTests(unittest.TestCase):
    def test_remote_history_preserves_cost_and_checks_endpoint(self):
        from harness_improvement import iterate_multi_agent_prompt as rhi
        expected = dict(comparison_key="0:1", prev_version=0, current_version=1,
                        prev_repo=ROOT, current_repo=ROOT, llm_backend="openrouter",
                        model="fixture-model", base_url="https://openrouter.ai/api/v1")
        row = dict(key="0:1", version_a=0, version_b=1, repo_a=str(ROOT), repo_b=str(ROOT),
                   llm_backend="openrouter", model="fixture-model", base_url=expected["base_url"])
        for cost in (0.012, None):
            row["call_metrics"] = {"estimated_cost_usd": cost}
            self.assertEqual(rhi._adjacent_history_provenance_errors(row, **expected), [])
        row["base_url"] = "https://other.example/v1"
        self.assertIn("base_url", rhi._adjacent_history_provenance_errors(row, **expected)[0])

    def test_final_comparison_retains_winner_without_another_proposal(self):
        self.run_final_comparison("local-vllm")

    def test_remote_optimizer_and_inherited_judge(self):
        self.run_final_comparison("openrouter")

    def test_remote_explicit_judge(self):
        self.run_final_comparison("openrouter", explicit_judge=True)

    def test_remote_champion_arbitration_checks_context_before_request(self):
        from harness_improvement import iterate_multi_agent_prompt as rhi
        with patch.object(rhi, "build_pairwise_user_message",
                          side_effect=["compare baseline", "compare candidate", "X" * 600000]):
            with self.assertRaisesRegex(SystemExit, "context guard.*champion"):
                self.run_final_comparison("openrouter")

    def run_final_comparison(self, backend_name, explicit_judge=False):
        from harness_improvement import iterate_multi_agent_prompt as rhi
        calls = []
        settings = []
        class FakeBackend:
            def __init__(self, **kwargs):
                self.backend = kwargs["backend"]
                settings.append(kwargs)
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
                    "--llm-backend", backend_name, "--model", "fixture-model", "--evaluate-only"]
            if backend_name == "local-vllm":
                argv += ["--require-local-vllm"]
            else:
                argv += ["--base-url", "https://openrouter.ai/api/v1", "--reasoning-effort", "medium",
                         "--max-output-tokens", "8192", "--context-window-tokens", "131072"]
                if explicit_judge:
                    argv += ["--judge-llm-backend", "openrouter", "--judge-model", "fixture-model",
                             "--judge-base-url", "https://openrouter.ai/api/v1",
                             "--judge-reasoning-effort", "medium"]
            with patch.object(sys, "argv", argv), patch.object(rhi, "JsonLLMBackend", FakeBackend), \
                 patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture-secret"}):
                self.assertEqual(rhi.main(), 0)
            retained = read_json(out / "retained_workflow.json")
            self.assertEqual(retained["version"], 1)
            self.assertTrue(retained["beats_reference"])
            self.assertEqual(len(calls), 3)  # baseline-v0, baseline-v1, champion arbitration
            self.assertFalse((out / "updated_queries/query51_ourTeam-v2.txt").exists())
            for config in settings:
                self.assertEqual(config["backend"], backend_name)
                if backend_name == "openrouter":
                    self.assertEqual(config["base_url"], "https://openrouter.ai/api/v1")
                    self.assertEqual(config["api_key"], "fixture-secret")
                    self.assertEqual(config["max_output_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
