---
name: mlip-deployment
description: "Export DPA4 or GRACE and validate energy, force and stress equivalence between the training runtime, ASE and native LAMMPS on UMT."
---

For general LAMMPS command checks, classical potentials and trajectory analysis, use [lammps](../lammps/SKILL.md). Export/runtime acceptance and the user-wide DPA4 inference policy remain in this skill; a generic accelerator recipe does not replace them.

## User-wide availability and project settings

Use this skill when the task matches its description in any project. Read and honor the current project's AGENTS.md, scientific brief and existing configuration. Do not carry a previous project's chemistry, software paths, run IDs, allocations or budgets into another project as defaults.

For cluster work, first resolve the current project's connection and configuration: `.mlip/config.json`, `configs/cluster.json` with its SSH helper/profile, or a configuration explicitly supplied for the task. Prefer an existing usable project setup. References to `.mlip/config.json` below mean the active project profile when an older project uses a different layout. Do not initialize a second profile merely because `.mlip` is absent.

Locate the collection checkout from the absolute path in `collection-root.txt` beside this installed SKILL.md, or from `MYSKILLS_ROOT`. In a checkout, go up two directories from this skill's folder (from `skills/<name>/` to the repository root). Read `toolkit/README.md` there. Resolve `recipes/`, `environments/`, `templates/` and `docs/` under `toolkit/`, or under the current project's installed `.mlip/`.

For a new project requiring a cluster profile, run `python toolkit/tools/setup.py --project "/path/to/project"` from the collection checkout. Select the user's UMT login for each new run or batch before launching it. A previous choice or a saved profile does not establish a default. The initializer builds a local profile using the explicitly supplied login, Slurm account and filesystem paths.

Read the reference and pin the trained checkpoint and exported model hashes. Compare identical structures in the native runtime, ASE and actual LAMMPS, including triclinic cells and periodic images. Declare tolerances and unit/sign/component mappings before comparison. Use the validated runtime activation and required plugin. Do not silently replace a CUDA major-version library with a symlink or assume the pip LAMMPS executable is suitable. Report serialization equivalence separately from DFT accuracy and MD stability. Revalidate newer export schemas, versions and multi-rank execution.

Read [workflow reference](references/workflow.md) for cluster-specific details and source recipes.

Read only the references and source files needed for the current task. Historical .source snapshots are recipes and evidence, not authorization to replay old jobs. Prefer the current project’s maintained implementation and retain its validation requirements.

## DPA4 inference policy for all projects

User instruction, 2026-11-12: use Triton-enabled frozen DPA4 models for repeated inference in every project.

- Export the chosen checkpoint to `.pt2` with `DP_TRITON_INFER=2` set during `dp --pt freeze -c CHECKPOINT.pt -o MODEL.pt2`. Set `DP_TF32_INFER=0` and `DP_AMP_INFER=0` for the validated float32 path. Changing Triton settings only at load time does not rebuild an existing export.
- Actually load the accepted `.pt2` in repeated evaluation, phonon-force calculations and ASE/MD inference. Creating and checking an export while continuing to evaluate the original `.pt` does not satisfy this rule. Retain `.pt` checkpoints and optimizer state for training, fine-tuning and restart; native training validation and explicit eager/export diagnostics may use `.pt`.
- Before accepting an export, compare energy, forces and stress against the source checkpoint on identical representative structures with declared tolerances and unit/sign conventions. Record checkpoint/export hashes, export flags, software versions and target GPU. Native LAMMPS consumption requires its own equivalence check. Re-export/revalidate for changed checkpoints or incompatible runtime/hardware.
- Allow sufficient compilation time: Neo/Triton2 took 514 seconds on the tested A100, so use a 20-minute freeze subprocess limit as the initial A100 Neo allowance, within the authorized Slurm walltime and allocation. A timeout or failed comparison rejects the artifact. Do not silently fall back to eager `.pt` or Triton0; diagnose and document a necessary exception explicitly. This policy does not authorize new cluster allocations by itself.
- Historical validation on the source A100 passed the energy, force and stress comparison on four structures. This establishes the tested inference path, not universal DFT accuracy or MD stability.
