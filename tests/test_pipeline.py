"""Offline invariants and stage integration using synthetic workspace fixtures."""
import ast
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mass import pipeline as p
from mass.io import ROOT, read_json, write_json
from mass.ranking import bradley_terry


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.c = read_json(ROOT / "configs/paper.json")
        self.c["run_dir"] = str(self.base)

    def test_configs_keep_model_generations_and_splits_separate(self):
        for name in ("paper.json", "cycle2.json"):
            c = p.validate(read_json(ROOT / "configs" / name))
            self.assertEqual(len(c["tasks"]["train"]), 8)
            self.assertEqual(c["collection"]["validation_rank"], 16)
        self.c["tasks"]["train"].append(60)
        with self.assertRaises(ValueError):
            p.validate(self.c)

    def test_saved_configuration_cannot_be_silently_changed(self):
        p.freeze_config(self.c)
        self.c["model_id"] = "another-model"
        with self.assertRaises(ValueError):
            p.freeze_config(self.c)

    def test_tournament_balanced_connected_and_unique(self):
        names = [f"e{i}" for i in range(18)]
        pairs = p.tournament_pairs(names, 6, 42)
        self.assertEqual(len(pairs), 54)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 54)
        self.assertEqual(set(collections.Counter(x for pair in pairs for x in pair).values()), {6})
        reached = {names[0]}
        while True:
            before = len(reached)
            for a, b in pairs:
                if a in reached or b in reached:
                    reached.update([a, b])
            if len(reached) == before:
                break
        self.assertEqual(reached, set(names))

    def test_bt_order_and_tie_symmetry(self):
        scores = bradley_terry(["a", "b", "c"], {("a", "b"): {"j": "a"}, ("a", "c"): {"j": "a"}, ("b", "c"): {"j": "b"}})
        self.assertGreater(scores["a"], scores["b"])
        self.assertGreater(scores["b"], scores["c"])
        tie = bradley_terry(["a", "b"], {("a", "b"): {"j": "tie"}})
        self.assertAlmostEqual(tie["a"], tie["b"])

    def test_selection_uses_rank16_and_never_test_tasks(self):
        for q in self.c["tasks"]["train"]:
            write_json(self.base / f"search/query{q}_ourTeam/retained_workflow.json", {"beats_reference": True})
            write_json(self.base / f"selection/task{q}_ranked.json", [{"episode": f"q{q}_r{i}", "rank": i, "task": q} for i in range(1, 19)])
        p.select(self.c)
        train = read_json(self.base / "selection/train.json")
        val = read_json(self.base / "selection/val.json")
        self.assertEqual(len(train), 120)
        self.assertEqual(len(val), 8)
        self.assertEqual({r["rank"] for r in val}, {16})
        self.assertFalse({r["episode"] for r in train} & {r["episode"] for r in val})
        self.assertFalse({r["task"] for r in train + val} & set(self.c["tasks"]["test"]))

    def test_checkpoint_selection_ignores_initial_and_unsaved_losses(self):
        out = self.base / "training"
        out.mkdir()
        rows = [{"step": 0, "val_loss": .1, "val_tokens": 20},
                {"step": 150, "val_loss": 2.0, "val_tokens": 20},
                {"step": 300, "val_loss": 1.0, "val_tokens": 20},
                {"step": 450, "val_loss": .2, "val_tokens": 20}]
        (out / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        for step in [150, 300]:
            d = out / f"step_{step:05d}"
            d.mkdir()
            # A zero-byte sentinel is sufficient: this test never loads weights.
            (d / "lora_state_dict.safetensors").touch()
        self.assertEqual(p.best_checkpoint(out)["step"], 300)

    def test_synthetic_search_collection_rank_select_integration(self):
        c = copy.deepcopy(self.c)
        c["tasks"] = {"train": [51], "train_candidates": [51], "excluded": [], "test": []}
        c["search"]["iterations"] = 2
        commands = []

        def fake_run(cmd, env=None):
            cmd = list(map(str, cmd))
            commands.append(cmd)
            if cmd[0] == "bash":
                prompt, ws, logs = map(Path, cmd[2:5])
                ws.mkdir(parents=True)
                logs.mkdir(parents=True)
                (logs / "prompt.txt").write_text(prompt.read_text())
                write_json(logs / "run_status.json", {"exit_code": 0, "trace_has_success_result": True})
            else:
                k = int(cmd[cmd.index("--current-version") + 1])
                out = Path(cmd[cmd.index("--out-dir") + 1])
                if "--evaluate-only" in cmd:
                    write_json(out / "retained_workflow.json", {"version": 1, "beats_reference": True})
                else:
                    q = Path(cmd[cmd.index("--query-file") + 1])
                    dest = out / f"updated_queries/query51_ourTeam-v{k+1}.txt"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(q.read_text())

        with patch.object(p, "run", fake_run):
            p.search(c, [51])
            p.collect(c, [51])
        with patch("mass.judging.self_judge", return_value={"winner": "A", "rationale": "synthetic fixture"}):
            p.rank(c, [51])
        p.select(c)
        self.assertEqual(len(read_json(self.base / "selection/train.json")), 15)
        self.assertEqual(len(read_json(self.base / "selection/val.json")), 1)
        self.assertEqual(sum("--evaluate-only" in cmd for cmd in commands), 1)
        self.assertTrue(all("--judge-model" not in cmd for cmd in commands))

    def test_windows_mask_context_and_train_each_assistant_token_once(self):
        # Load the exact pure function without importing Transformers/PyTorch.
        source = (ROOT / "training/prepare_sft_data.py").read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "make_windows")
        scope = {"IGNORE": -100}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "renderer_function", "exec"), scope)
        def block(kind, ids, loss):
            return {"kind": kind, "ids": ids, "mask": [loss] * len(ids)}
        blocks = [block("system", [1, 2], False), block("user", [3, 4], False),
                  block("assistant", [10, 11, 12], True), block("tool", [20, 21], False),
                  block("assistant", [30, 31, 32], True)]
        windows, skipped = scope["make_windows"](blocks, window=10, overlap=3)
        self.assertFalse(skipped)
        self.assertGreater(len(windows), 1)
        self.assertEqual([x for w in windows for x in w["labels"] if x != -100], [10, 11, 12, 30, 31, 32])
        self.assertTrue(all(len(w["input_ids"]) <= 10 for w in windows))


if __name__ == "__main__":
    unittest.main()
