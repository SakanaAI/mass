# Synthetic research tasks

This folder includes the full prompts for all 12 synthetic tasks:
nine designated training tasks and three test tasks, as in the paper.
Each task also includes its initial workflow for RHI.

The split refers to post-training. Workflow search can be evaluated on all 12
tasks, but the three test tasks do not enter SFT, its validation set, or
checkpoint selection.

## Nine designated training tasks

| Task | Domain | Study | Full task prompt | Initial workflow |
|---|---|---|---|---|
| 51 | Finance | Macro signals and cross-asset trading | [Prompt](bare/query51.txt) | [Task + workflow](initial/query51_ourTeam.txt) |
| 53 | Finance | Trading around monetary-policy announcements | [Prompt](bare/query53.txt) | [Task + workflow](initial/query53_ourTeam.txt) |
| 56 | Finance | Yield-curve relative-value strategies | [Prompt](bare/query56.txt) | [Task + workflow](initial/query56_ourTeam.txt) |
| 202 | Pharmacy | Protein-interface prediction | [Prompt](bare/query202.txt) | [Task + workflow](initial/query202_ourTeam.txt) |
| 205 | Pharmacy | Molecular-dynamics force-field validation | [Prompt](bare/query205.txt) | [Task + workflow](initial/query205_ourTeam.txt) |
| 221 | Pharmacy | Protein-sequence tokenization | [Prompt](bare/query221.txt) | [Task + workflow](initial/query221_ourTeam.txt) |
| 302 | Robotics | Learned heuristics for path planning | [Prompt](bare/query302.txt) | [Task + workflow](initial/query302_ourTeam.txt) |
| 307 | Robotics | Synthetic manipulation-planning datasets | [Prompt](bare/query307.txt) | [Task + workflow](initial/query307_ourTeam.txt) |
| 324 | Robotics | Subgoal discovery for navigation | [Prompt](bare/query324.txt) | [Task + workflow](initial/query324_ourTeam.txt) |

In the reported experiment, workflow search did not
find a workflow for task 221 that beat its reference, so it was excluded from
teacher collection. The other eight tasks contributed SFT data. This exclusion
does not move task 221 into the test split.

## Three test tasks

| Task | Domain | Study | Full task prompt | Initial workflow |
|---|---|---|---|---|
| 60 | Finance | Trading execution costs and capacity | [Prompt](bare/query60.txt) | [Task + workflow](initial/query60_ourTeam.txt) |
| 207 | Pharmacy | Inverse-folding evaluation | [Prompt](bare/query207.txt) | [Task + workflow](initial/query207_ourTeam.txt) |
| 305 | Robotics | Tool-use planning for manipulation | [Prompt](bare/query305.txt) | [Task + workflow](initial/query305_ourTeam.txt) |

## Use the tasks

[splits.json](splits.json) lists the nine designated training tasks, three test
tasks, and the eight tasks used for SFT. The corresponding fields in both
cycle configurations are:

| Configuration field | Meaning |
|---|---|
| `tasks.train_candidates` | All nine designated training tasks |
| `tasks.train` | The eight tasks used for SFT in the paper configuration |
| `tasks.test` | The three test tasks |
| `tasks.excluded` | Task 221, excluded from teacher collection |

The default `search` stage includes all 12 tasks. To run RHI on the nine
designated training tasks only, run this from the repository root after setup:

```bash
python -m mass search --config configs/paper.json \
  --tasks 51 53 56 202 205 221 302 307 324
```

Continue with the collection and post-training commands in
[running.md](../docs/running.md). They use the eight SFT tasks specified in the
paper configuration. The [teacher directive](teacher_directive.txt) is appended
to a retained task–workflow prompt during teacher collection.

The task prompts specify the public data sources and required deliverables.
Large task datasets are obtained when running the tasks and are not bundled in
this lightweight release.
