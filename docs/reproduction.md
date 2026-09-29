# Implementation notes

The repository combines workflow search, trajectory selection, conversation
rendering, LoRA training, and evaluation in a stage-based driver. The
[paper-to-code mapping](paper_to_code.md) describes the notation and training
settings; [experiment support](experiment_coverage.md) lists the available
experiments and analyses.

## Workflow search

The current RHI driver uses a fixed bare-task reference. Proposals are compared
against that reference, and eligible winners are compared with the retained
champion. Algorithm 1 in the paper instead compares each proposal directly with
the retained best output. These rules can select different workflows, so the
current driver is not an exact implementation of Algorithm 1.

The search command records proposed workflows, workspaces, comparisons, and the
retained workflow. The final comparison does not generate another proposal.

## Training configurations

The supplied configurations use the current model for teacher ranking, rank 16
for validation, and minimum validation loss for checkpoint selection. Earlier
experimental branches also used external ranking or newly collected validation
trajectories; those settings are not used by these configurations. The repository
does not contain the complete run ledger linking every reported result to its
historical configuration.

The first-cycle hyperparameters are recorded in the paper and
`configs/paper.json`. The second cycle reuses this recipe, including a
1,624-step budget. That second-cycle budget is a reconstruction choice; the
exact final data and budget manifest is not included.

Within each task, a seeded connected tournament gives each candidate six
opponents, followed by Bradley–Terry ranking. The schedule and seeds are
reconstruction choices, rather than the original tournament ledger.

Fresh runs save their actual trajectory counts, token windows, and validation
losses. The reported first-cycle count of 119 training trajectories reflects
one premature termination; a new run can produce a different valid count and
select a different checkpoint.

## Synthetic evaluation

Three current behaviors matter when choosing an evaluation pool:

- `rollout --split all` includes all 12 tasks, including task 221. The paper's
  pooled task-solving result uses eight SFT tasks and three test tasks. Pass
  those 11 task IDs explicitly to evaluate that pool.
- Each rollout call replaces `evaluation/rollouts.json`. Use a separate run
  directory for each evaluation pool or save the manifest before another call.
- `report` summarizes all judgments stored in that run directory's
  `evaluation/external.jsonl`, including earlier calls. Use a separate reporting
  run directory for each comparison set.

The pairwise evaluator accepts workspace pairs supplied by the researcher.
The paper's selected fixed-reference lists and the success/0.6-deliverable
eligibility filter used for six tasks are not reconstructed by this driver.

Completed episodes are reused; incomplete episode directories are preserved.
The synthetic runner stops on a nonzero process exit. It does not automate the
full distinction between retained loop halts and retried infrastructure
failures. Inspect a failed episode before retrying in a new run directory.
The public benchmark launcher has its own retry mechanism.

## Public benchmarks and run records

Public model and benchmark source revisions are pinned. Model sampling, live
task data, and external judge services can still change results. Keep the run
configuration, generated workspaces, scoring logs, and environment details
with each experiment.

The MLR-Bench image recipe contains unpinned scientific Python packages.
Record the built image digest and package list; a rebuilt image is not an
exact copy of the historical environment.

ScienceAgentBench summaries use the official evaluator's per-instance rows and
require 102 rows per trial. The original instance-level logs and within-trial
aggregation behind the paper's supplied three-trial summaries are not included.
The scripts compute new summaries from evaluator output; they do not use the
paper's aggregate values as test expectations.

## Validation

The offline tests cover ranking, split isolation, token masking, checkpoint
selection, benchmark aggregation, and pipeline stages with synthetic fixtures.
The RHI integration test uses a mocked model backend. These checks do not
reproduce experimental scores. The pinned core dependencies installed successfully
in a fresh Python 3.13 environment, and all 12 offline tests passed on
28 September 2026. Full training/serving dependency installation, GPU training,
and end-to-end benchmark reruns have not been validated for this package.
