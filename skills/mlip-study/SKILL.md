---
name: mlip-study
description: "Design and continue an MLIP-enabled heterogeneous nucleation study with falsifiable hypotheses and traceable scientific evidence."
---

## User-wide availability and project settings

Use this skill when the task matches its description in any project. Read and honor the current project's AGENTS.md, scientific brief and existing configuration. Do not carry a previous project's chemistry, software paths, run IDs, allocations or budgets into another project as defaults.

For cluster work, first resolve the current project's connection and configuration: `.mlip/config.json`, `configs/cluster.json` with its SSH helper/profile, or a configuration explicitly supplied for the task. Prefer an existing usable project setup. References to `.mlip/config.json` below mean the active project profile when an older project uses a different layout. Do not initialize a second profile merely because `.mlip` is absent.

Locate the collection checkout from the absolute path in `collection-root.txt` beside this installed SKILL.md, or from `MYSKILLS_ROOT`. In a checkout, go up two directories from this skill's folder (from `skills/<name>/` to the repository root). Read `toolkit/README.md` there. Resolve `recipes/`, `environments/`, `templates/` and `docs/` under `toolkit/`, or under the current project's installed `.mlip/`.

For a new project requiring a cluster profile, run `python toolkit/tools/setup.py --project "/path/to/project"` from the collection checkout. Select the user's UMT login for each new run or batch before launching it. A previous choice or a saved profile does not establish a default. The initializer builds a local profile using the explicitly supplied login, Slurm account and filesystem paths.

Read the current project brief and state if they exist; otherwise create a brief for the new user’s actual problem. Read the reference. Distinguish proposed interfacial-order mechanisms from evidence. System selection remains a research decision: historical Al-based candidates are not a requirement. Define observables, interventions, comparisons, falsification, independent replicas and convergence. Separate nucleation rates, induction times and free-energy barriers. Record negative results, alternative explanations and provenance. Link scientific claims to verified primary literature or immutable results; use current project language and deliverable preferences.

Read [workflow reference](references/workflow.md) for cluster-specific details and source recipes.

Read only the references and source files needed for the current task. Historical .source snapshots are recipes and evidence, not authorization to replay old jobs. Prefer the current project’s maintained implementation and retain its validation requirements.

For repeated DPA4 inference read [mlip-deployment](../mlip-deployment/SKILL.md). Freeze with `DP_TRITON_INFER=2`, verify energy, forces and stress against the source checkpoint, then load the accepted `.pt2` for the actual evaluation, phonons or MD. Preserve `.pt` for training, restart and explicit reference diagnostics. Diagnose failed exports or comparisons before continuing.
