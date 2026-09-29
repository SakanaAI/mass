"""Aggregate the official-rubric reviews, requiring both judges and all tasks."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def summarize(reviews, expected_tasks):
    groups = collections.defaultdict(dict)
    for row in reviews.values():
        if row["task"] not in expected_tasks:
            raise ValueError(f"Unexpected task: {row['task']}")
        judges = row["reviews"]
        if set(judges) != {"gpt-5.5", "claude-opus-4-8"}:
            raise ValueError(f"Missing judge for {row['job']}/{row['task']}")
        scores = [float(v["Overall"]) for v in judges.values()]
        for value in judges.values():
            score = float(value["Overall"])
            if value.get("missing_paper"):
                if score != 0:
                    raise ValueError("A missing paper must receive zero")
            elif not 1 <= score <= 10:
                raise ValueError("A present paper must have Overall between 1 and 10")
        if row["task"] in groups[row["job"]]:
            raise ValueError("Duplicate task in trial")
        groups[row["job"]][row["task"]] = statistics.mean(scores)
    players = collections.defaultdict(dict)
    for job, tasks in groups.items():
        if set(tasks) != set(expected_tasks):
            raise ValueError(f"Incomplete MLR trial: {job}")
        _, generation, trial = job.split("_")
        players[generation][trial] = statistics.mean(tasks.values())
    if not players:
        raise ValueError("No reviews")
    return {g: {"trial_means": trials, "mean": statistics.mean(trials.values()),
                "sample_sd": statistics.stdev(trials.values()) if len(trials) > 1 else None}
            for g, trials in players.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reviews", type=Path)
    a = p.parse_args()
    names = (Path(__file__).resolve().parent / "configs/mlr_tasks.txt").read_text().splitlines()
    print(json.dumps(summarize(json.loads(a.reviews.read_text()), {f"mlr_{n}" for n in names}), indent=2))
