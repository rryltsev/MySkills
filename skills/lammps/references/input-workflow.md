# LAMMPS input, runtime and physical validation

Use for preparing/reviewing inputs, selecting classical potentials, or diagnosing execution. This is a decision guide; exact syntax and package support come from documentation matching the installed version. The original supplied LAMMPS skill was generalized on 2026-11-18; its fixed hardware/performance examples are not defaults.

## Potential and structure

- Choose a particular parameterization, not merely a potential family. Check supported elements, cross interactions, phase, temperature/pressure/deformation range and relevant validation. EAM/FS/MEAM, Tersoff/SW, AIREBO and ReaxFF are candidates, not automatic recommendations for an element. Treat a user-requested idealized/toy model as such rather than imposing a materials benchmark on it.
- Preserve potential filename, checksum, source/publication, version, units and element-to-type mapping. The [NIST repository](https://www.ctcms.nist.gov/potentials/), authors' supplements and LAMMPS distribution are useful sources; presence there alone is not validation for the current question.
- Check `units`, `dimension`, `atom_style`, masses, charges/topology where relevant, box/PBC, minimum separations and `pair_coeff` order together. Do not merge independent elemental potentials into an alloy model without defined cross interactions. Reactive and molecular models may need charge equilibration, bonded terms, exclusions and long-range electrostatics; inspect their own documentation.
- MLIP export and stress/virial capability belong to [mlip-deployment](../../mlip-deployment/SKILL.md); repeated DPA4 inference must consume its accepted Triton2 `.pt2` artifact.

## Runtime and accelerator choice

Record the exact executable/version and `lmp -h` packages/styles. A conda/pip installation does not guarantee the required GPU backend, MLIP plugin or MPI compatibility. Require only packages actually used by the input; do not universally require GPU, MANYBODY or EXTRA-COMPUTE.

Match the MPI launcher with the MPI implementation linked into LAMMPS. The active environment's launcher is a candidate, not proof of compatibility; scheduler-provided `srun` may be correct when PMI/PMIx and site configuration match. If ranks exist but logs never appear, inspect launcher/library compatibility and a small startup test before changing binding flags or increasing rank counts. Cluster tests use the current project's allocation policy; UMT-specific mechanics remain in [umt-hpc](../../umt-hpc/SKILL.md).

Choose GPU, KOKKOS, OPENMP, serial or the MLIP plugin according to the actual style/build. A `/gpu` suffix only accelerates supported styles. `package gpu 4` is not a generic way to enable MLIP inference, and four GPUs are not a default. Confirm allocated devices, rank/thread counts, precision and measured throughput. Compare observables within declared numerical tolerances when changing backend. See [accelerator packages](https://docs.lammps.org/Speed_packages.html) and [GPU package](https://docs.lammps.org/Speed_gpu.html).

## Ensemble, loading and timestep

Keep an explicit stage sequence: preparation/minimization if appropriate, equilibration, loading or production, output and restart. Ensure each mobile atom has the intended integration once; for example `fix nvt`/`npt` already integrate their group. Thermostat/barostat time constants are in the selected time units. Establish timestep stability for the potential, temperature and fastest motion; 0.001 ps in `metal` is an example, not a universal choice. Inspect drift in an appropriate conservation test separately from expected thermostat or external-work energy exchange.

NPT is appropriate only for the intended ensemble and compatible box degrees of freedom. Free surfaces, constrained directions and non-equilibrium flows require their own treatment. Do not impose NPT on every system or simultaneously control a box component by incompatible fixes. Consult [Nosé-Hoover fixes](https://docs.lammps.org/fix_nh.html).

For deformation, define loading direction, engineering versus true strain/rate, box tilt, coordinate/velocity remapping, transverse constraints and the temperature definition. Separate streaming motion from thermal motion where required; check compatibility of thermostat and `fix deform` settings. State the stress sign, units and volume normalization, especially for slabs/nanowires. See [fix deform](https://docs.lammps.org/fix_deform.html).

For shock, define piston/impact/loading method, transverse conditions, sample length and observation time before wave reflections. Choose `p`, `f`, `s` or `m` from this geometry: shrink-wrapping is not a universal shock boundary, and fixed open faces can lose atoms. Specialized periodic shock methods need their own documented assumptions. Preserve spatial density/temperature/velocity/stress profiles and the frame of reference. See [boundary](https://docs.lammps.org/boundary.html).

## Output and acceptance

Preserve input/data/potential hashes, software/runtime, type map, step/time and seeds. Trajectories need stable IDs, types, coordinates, cell/tilts and PBC; diffusion requires sufficient unwrapping information. Save velocities, per-atom quantities or full pressure time series when the requested observable needs them. Do not substitute rendered images or averaged RDF for raw fluctuation data. Exact restart needs the relevant integrator/fix state as well as coordinates; consult [read_restart](https://docs.lammps.org/read_restart.html).

An initialization/`run 0` check detects setup failures; a short pilot assesses force magnitude, finite energies, temperature/pressure/volume, phase and stability. Production acceptance also checks intended physical duration, sampling and statistical precision. Do not accept missing atoms, NaNs or unexpected phase changes merely because the process exited successfully. For MLIP ensembles use [ase-lammps-md](../../ase-lammps-md/SKILL.md), for thermodynamics and transport use [liquid-properties](../../liquid-properties/SKILL.md).

Structural outputs should answer the material question: CNA/PTM phase fractions, centrosymmetry/defects, Voronoi statistics, dislocations, or chemical order as appropriate. Inspect algorithm thresholds and temperature/strain sensitivity; a structure label alone is not phase proof. Visualizations should include a legend, time/state and the selection/cutoff definition. Use the [OVITO reference](ovito-analysis.md) for processing mechanics and the [RDF/Cowley reference](rdf-cowley-analysis.md) for chemical neighborhoods.
