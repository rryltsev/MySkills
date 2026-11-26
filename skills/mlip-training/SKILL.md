---
name: mlip-training
description: "Train and fine-tune DPA4/DPA4c or GRACE, preserve frozen dataset splits, select checkpoints on validation and resume optimizer state."
---

## User-wide availability and project settings

Use this skill when the task matches its description in any project. Read and honor the current project's AGENTS.md, scientific brief and existing configuration. Do not carry a previous project's chemistry, software paths, run IDs, allocations or budgets into another project as defaults.

For cluster work, first resolve the current project's connection and configuration: `.mlip/config.json`, `configs/cluster.json` with its SSH helper/profile, or a configuration explicitly supplied for the task. Prefer an existing usable project setup. References to `.mlip/config.json` below mean the active project profile when an older project uses a different layout. Do not initialize a second profile merely because `.mlip` is absent.

Locate the collection checkout from the absolute path in `collection-root.txt` beside this installed SKILL.md, or from `MYSKILLS_ROOT`. In a checkout, go up two directories from this skill's folder (from `skills/<name>/` to the repository root). Read `toolkit/README.md` there. Resolve `recipes/`, `environments/`, `templates/` and `docs/` under `toolkit/`, or under the current project's installed `.mlip/`.

For a new project requiring a cluster profile, run `python toolkit/tools/setup.py --project "/path/to/project"` from the collection checkout. Select the user's UMT login for each new run or batch before launching it. A previous choice or a saved profile does not establish a default. The initializer builds a local profile using the explicitly supplied login, Slurm account and filesystem paths.

Read the reference, the dataset manifest, model metadata and user profile. Use exact checkpoint hashes and compatible backends. Freeze grouped train/validation/test membership before fitting. Keep held-out test labels outside training/selection payloads. Fit any energy-reference correction on train only. Include the starting foundation checkpoint among candidates when comparing fine-tuning. Select only on the predefined validation score; lock the selection before test evaluation. For restart inspect the actual optimizer counter and loop index; use the retained tested Adam workaround only for its matching DeePMD version/setup. Synthetic checks establish software operation, not physical model accuracy.

Read [workflow reference](references/workflow.md) for cluster-specific details and source recipes.

Read only the references and source files needed for the current task. Historical .source snapshots are recipes and evidence, not authorization to replay old jobs. Prefer the current project’s maintained implementation and retain its validation requirements.

For DPA4 downstream repeated inference, follow the user-wide Triton2 freeze policy in [mlip-deployment](../mlip-deployment/SKILL.md): export and consume a validated `.pt2`. Preserve native `.pt` checkpoints and optimizer state for training/restart and native training validation.
