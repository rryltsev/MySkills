---
name: mlip-environments
description: "Set up isolated DPA4/DPA4c and GRACE environments on UMT, restore pinned GRACE packages and diagnose CUDA, Keras and filesystem issues."
---

## User-wide availability and project settings

Use this skill when the task matches its description in any project. Read and honor the current project's AGENTS.md, scientific brief and existing configuration. Do not carry a previous project's chemistry, software paths, run IDs, allocations or budgets into another project as defaults.

For cluster work, first resolve the current project's connection and configuration: `.mlip/config.json`, `configs/cluster.json` with its SSH helper/profile, or a configuration explicitly supplied for the task. Prefer an existing usable project setup. References to `.mlip/config.json` below mean the active project profile when an older project uses a different layout. Do not initialize a second profile merely because `.mlip` is absent.

Locate the collection checkout from the absolute path in `collection-root.txt` beside this installed SKILL.md, or from `MYSKILLS_ROOT`. In a checkout, go up two directories from this skill's folder (from `skills/<name>/` to the repository root). Read `toolkit/README.md` there. Resolve `recipes/`, `environments/`, `templates/` and `docs/` under `toolkit/`, or under the current project's installed `.mlip/`.

For a new project requiring a cluster profile, run `python toolkit/tools/setup.py --project "/path/to/project"` from the collection checkout. Select the user's UMT login for each new run or batch before launching it. A previous choice or a saved profile does not establish a default. The initializer builds a local profile using the explicitly supplied login, Slurm account and filesystem paths.

Read `.mlip/config.json` and the reference. Prefer an accessible, administrator-approved installation or an independently installed prefix owned by the current user. Preserve existing environments. Keep GRACE separate from DeePMD. Restore GRACE from the bundled exact Linux locks when appropriate. DPA4 package versions are an observed recipe, not a full reproducibility lock. Run installation, imports that initialize accelerators, and training checks under Slurm. Put installer HOME and all caches on compute-writable storage using a child process or subshell. Verify actual GPU use and finite optimizer updates before qualifying the environment.

Read [workflow reference](references/workflow.md) for cluster-specific details and source recipes.

Read only the references and source files needed for the current task. Historical .source snapshots are recipes and evidence, not authorization to replay old jobs. Prefer the current project’s maintained implementation and retain its validation requirements.

For repeated DPA4 inference read [mlip-deployment](../mlip-deployment/SKILL.md). Freeze with `DP_TRITON_INFER=2`, verify energy, forces and stress against the source checkpoint, then load the accepted `.pt2` for the actual evaluation, phonons or MD. Preserve `.pt` for training, restart and explicit reference diagnostics. Diagnose failed exports or comparisons before continuing.
