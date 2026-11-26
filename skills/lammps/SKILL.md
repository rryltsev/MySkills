---
name: lammps
description: "Prepare, review and troubleshoot LAMMPS inputs and classical potentials; analyze trajectories with OVITO, RDF, coordination and Warren-Cowley chemical order. Use across materials and projects; MLIP deployment and liquid transport have companion skills."
---

# LAMMPS: inputs and structural analysis

Use in any computational project needing LAMMPS or compatible atomistic trajectory analysis, including classical and machine-learned potentials. Honor the current project's AGENTS.md, scientific question and existing environment. Chemistry, paths, hardware, ensemble, tolerances and allocation budgets are project-specific. Local analysis does not require a cluster profile.

## Scope and companion skills

Keep each procedure in one place; read companions only for the relevant task:

| Task | Owning skill or reference |
|---|---|
| LAMMPS commands, classical potential suitability, deformation, shock, structural outputs | This skill and [input workflow](references/input-workflow.md) |
| RDF shell boundaries, coordination, directed/local Warren-Cowley and uncertainty | [RDF/Cowley protocol](references/rdf-cowley-analysis.md) and [NumPy helpers](scripts/rdf_cowley.py) |
| Compressed trajectories, OVITO APIs, periodic neighborhoods, CPU parallelism and visualization | [OVITO workflow](references/ovito-analysis.md) |
| Reproducible MLIP NPT/NVT, independent replicas and restart semantics | [ase-lammps-md](../ase-lammps-md/SKILL.md) |
| DPA4/GRACE export, runtime and native/ASE/LAMMPS equivalence | [mlip-deployment](../mlip-deployment/SKILL.md) |
| Density, mixing enthalpy, diffusion, Green-Kubo viscosity and STACIE | [liquid-properties](../liquid-properties/SKILL.md) |
| SSH, Slurm submission/recovery and resources on UMT | [umt-hpc](../umt-hpc/SKILL.md) |

For **DPA4 inference**, read the policy in `mlip-deployment` before choosing a model path. Repeated inference must actually consume a validated `.pt2` exported with `DP_TRITON_INFER=2`; compare energy, forces and stress before acceptance. Retain `.pt` for training/restart and explicit reference diagnostics. Export or validation failure must not silently switch to eager inference or Triton0. A generic LAMMPS/GPU recipe does not override this policy.

## Working procedure

1. Inspect the inputs, data/type mapping, potential provenance, LAMMPS version, installed packages and executable/runtime. Use the current project's working setup where suitable.
2. Check version-matched [official LAMMPS documentation](https://docs.lammps.org/) for the commands and restrictions the task depends on. Optional local documentation indexes may help, but `scholaraio` is not a dependency. Resolve aliases such as `fix npt` → `fix_nh` and EAM styles → `pair_eam` yourself.
3. Choose the potential and protocol for the physical observable and state range. Record parameter sources. For strict reproduction, inspect accessible papers/supplements before identifying any missing parameter; label engineering choices separately.
4. Prepare a consistent input and output protocol using the [input workflow](references/input-workflow.md). Do not prescribe universal NPT equilibration, a 1 fs timestep, four GPUs or one shock boundary style.
5. For requested execution, validate initialization and a short stability/performance pilot within the active project's resources before production. Match MPI launcher/runtime and accelerator support to the actual executable. Input review alone does not authorize production jobs.
6. Preserve provenance and raw observables needed for analysis. Check physical stability, frame counts and units in addition to process success. Interpret structural or mechanical observables relevant to the question.

## Analysis essentials

Read the RDF/Cowley reference before estimating shell boundaries or chemical order. Distinguish RDF bin centers from cumulative-count upper edges, directed neighbor probabilities from symmetric pair distributions, and atom-weighted from coordination-weighted local averages. Use an explicitly defined common shell and baseline; propagate radius uncertainty when the radius is data-derived. A jumping selected bin is not evidence of a structural anomaly.

Read the OVITO reference when using that engine. Preserve IDs, species mapping, changing cell/PBC and units; validate numerical equivalence before optimizing throughput. Use independent trajectories or justified correlation-aware blocks for uncertainty, not a raw count of atoms or adjacent frames.

The NumPy module is an analysis helper, not a trajectory parser or an automatic shell selector. Run `python scripts/rdf_cowley.py` from this skill directory for its analytical checks. Resolve paths against the installed skill folder when calling from another project. OVITO and LAMMPS are optional dependencies only for workflows that use them; helper checks need NumPy.
