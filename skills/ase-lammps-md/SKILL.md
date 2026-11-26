---
name: ase-lammps-md
description: "Prepare and run reproducible MLIP molecular dynamics in ASE or LAMMPS, including NPT density and independent NVT transport replicas."
---

## Related workflows

For general LAMMPS input syntax, classical potentials, deformation/shock setup and OVITO/RDF/Cowley analysis, use [lammps](../lammps/SKILL.md). This skill owns MLIP ensemble preparation, replicas and restart semantics. Before DPA4 inference, read [mlip-deployment](../mlip-deployment/SKILL.md): repeated evaluation and MD must actually load the validated Triton2 frozen `.pt2`, with energy/force/stress equivalence verified; never silently substitute `.pt` or Triton0 after failure.

## User-wide availability and project settings

Use this skill when the task matches its description in any project. Read and honor the current project's AGENTS.md, scientific brief and existing configuration. Do not carry a previous project's chemistry, software paths, run IDs, allocations or budgets into another project as defaults.

For cluster work, first resolve the current project's connection and configuration: `.mlip/config.json`, `configs/cluster.json` with its SSH helper/profile, or a configuration explicitly supplied for the task. Prefer an existing usable project setup. References to `.mlip/config.json` below mean the active project profile when an older project uses a different layout. Do not initialize a second profile merely because `.mlip` is absent.

Locate the collection checkout from the absolute path in `collection-root.txt` beside this installed SKILL.md, or from `MYSKILLS_ROOT`. In a checkout, go up two directories from this skill's folder (from `skills/<name>/` to the repository root). Read `toolkit/README.md` there. Resolve `recipes/`, `environments/`, `templates/` and `docs/` under `toolkit/`, or under the current project's installed `.mlip/`.

For a new project requiring a cluster profile, run `python toolkit/tools/setup.py --project "/path/to/project"` from the collection checkout. Select the user's UMT login for each new run or batch before launching it. A previous choice or a saved profile does not establish a default. The initializer builds a local profile using the explicitly supplied login, Slurm account and filesystem paths.

Read the reference, the requested ensemble and `.mlip/config.json`. Freeze structures, species mapping, masses, checkpoint/export hashes, temperature, pressure, timestep, thermostat/barostat, seeds and sampling intervals in a run protocol. Derive model type indices from the full model type map; a Ti-Al fine-tuned checkpoint can still contain a periodic-table map. Verify CPU/GPU and ASE/LAMMPS predictions before production. Benchmark a small pilot on the allocated hardware and check its projected budget. Keep equilibration and production distinct, replicas identifiable and restart semantics explicit. Continue from a last frame only when an ensemble reinitialization is acceptable; that is not an exact integrator restart.

Read [workflow reference](references/workflow.md) for cluster-specific details and source recipes.

Read only the references and source files needed for the current task. Historical .source snapshots are recipes and evidence, not authorization to replay old jobs. Prefer the current project’s maintained implementation and retain its validation requirements.
