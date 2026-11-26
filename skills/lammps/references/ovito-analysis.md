# OVITO Python for efficient LAMMPS trajectory analysis

Use for trajectory post-processing, local structure, RDF/coordination, and custom chemical-neighborhood calculations. This reference complements [rdf-cowley-analysis.md](rdf-cowley-analysis.md); it does not replace its cutoff, normalization, uncertainty, or fluctuation rules. Prefer existing sufficient raw counts/moments over rereading coordinates. Checked against official OVITO 3.17.2 documentation on 2026-11-12. Examples below are templates, **not locally executed OVITO benchmarks**.

## Inspect the installed API first

The user-wide integration rechecked the core reader, RDF and multiprocessing documentation on 2026-11-18. This does not establish that OVITO is installed in a particular project; inspect that environment before choosing an API.

Record the actual interpreter and `ovito.version_string`; inspect required classes/properties before a production pass. Current documentation calls the RDF class `RadialDistributionFunctionModifier`; older workflows use `CoordinationAnalysisModifier`. Do not infer that an old alias is available from a cached documentation page. A conditional import is appropriate:

```python
import ovito
from ovito.io import import_file
try:
    from ovito.modifiers import RadialDistributionFunctionModifier as RDFModifier
except ImportError:
    from ovito.modifiers import CoordinationAnalysisModifier as RDFModifier
```

The changelog records the modifier's renaming; separately, it lists the addition of zstandard support in 3.11.1. Per-type coordination output is also version-dependent; feature-check the computed data. Do not install or upgrade packages solely to refresh this reference. [Official changelog](https://www.ovito.org/docs/current/new_features.html), [current Python API](https://docs.ovito.org/python/modules/ovito_modifiers.html#ovito.modifiers.RadialDistributionFunctionModifier).

## Compressed streams and archives are different

| Input | Confirmed behavior / action |
|---|---|
| Supported LAMMPS trajectory ending `.gz` | Native direct import; no pre-extraction required. |
| Supported LAMMPS trajectory ending `.zst` | Native direct import in documented versions; verify the installed version. |
| `.xz`, `.bz2` compressed trajectories | Native `import_file()` support was **not established** by the checked documentation. Test explicitly; otherwise prepare only the needed input in bounded scratch space. |
| `.tar`, `.tar.gz`, `.tar.zst`, `.zip`, `.7z` containers | Do not assume direct archive-member import. Inspect the container and extract only required members if necessary. A member that is itself `.dump.gz` can remain compressed. |

The LAMMPS reader supports a multi-frame file, a filename sequence, and an explicit list of files. It does not support LAMMPS `%` processor-split dump sets as a directly assembled trajectory. Preserve original archives and their provenance; any cleanup concerns only task-owned scratch after verified output. [Official LAMMPS reader](https://docs.ovito.org/reference/file_formats/input/lammps_dump.html), [data import](https://docs.ovito.org/usage/import.html).

```python
pipeline = import_file("trajectory.dump.gz")
# Also valid: import_file("frames/frame.*.dump.gz") or an explicit path list.
print(pipeline.num_frames)
for frame in range(pipeline.num_frames):
    data = pipeline.compute(frame)
    # Reduce this frame to required statistics and persist a bounded result.
```

`import_file()` discovers frames; evaluation loads frame data on demand. `pipeline.frames` is an alternative iterator. This avoids retaining the whole trajectory, but modifiers and caches still consume memory: lazy loading is not a fixed-memory guarantee. Reuse a pipeline and change its `source.load(...)` for compatible files; do not rebuild it for every frame. [File I/O](https://docs.ovito.org/python/introduction/scripting/file_io.html).

## RDF, coordination and chemical analysis

Append `RDFModifier(cutoff=6.0, number_of_bins=240, partial=True)`. Inspect `data.tables['coordination-rdf'].xy()` and `table.y.component_names` instead of assuming component ordering. Current outputs include particle `Coordination` and, with partial analysis, `Per Type Coordination`; use the latter only if present. [RDF API](https://docs.ovito.org/python/modules/ovito_modifiers.html#ovito.modifiers.RadialDistributionFunctionModifier).

OVITO's documented partials satisfy, for a binary mixture,

    g_total = x_A² g_AA + 2x_A x_B g_AB + x_B² g_BB.

`only_selected=True` removes unselected atoms from both neighbor counting and RDF normalization. It is therefore not a way to retain every neighbor while selecting only central atoms of one type. Use full-system partials or explicit neighbor queries for that purpose. [RDF normalization and selection](https://docs.ovito.org/reference/pipelines/modifiers/radial_distribution_function.html).

Project rules: verify OVITO's same-species finite-N normalization against raw pair counts before combining it with LAMMPS or an existing analysis. Retain instantaneous NPT volume and actual species counts. OVITO has no internal physical unit system and ignores LAMMPS ITEM: UNITS; carry the simulation unit convention explicitly. Distinguish ∫g d³r (volume units) from coordination; preserve directed-pair conventions, exact shell volumes, bin centers versus cumulative upper edges, pure-endpoint masks, and the accepted RDF-minimum method. Calculate per-frame integrals/chemical contrasts before variances; a time-averaged RDF cannot recover their fluctuations. Preserve local-atom, global-frame, within-replica and between-replica distinctions. Raw volume fluctuations remain physical observables.

## Custom neighborhoods without Python pair loops

`CutoffNeighborFinder(cutoff, data)` handles spherical neighborhoods and periodic cells. Its `find_all(indices=None, sort_by=None)` returns directed central/neighbor index pairs and displacement vectors in vectorized arrays. For huge systems, query center batches to bound pair-list memory. `NearestNeighborFinder(k, data)` instead chooses k neighbors; the documented maximum is 64. Do not substitute fixed-k neighborhoods for a physical radial cutoff.

Periodic images can repeat the same atom index, including self images in a small box. For a minimum-image RDF/count convention, enforce the applicable cutoff/half-cell-height restriction and verify self exclusion; do not deduplicate by atom index blindly. [Neighbor-finder API](https://docs.ovito.org/python/modules/ovito_data.html#ovito.data.CutoffNeighborFinder), [nearest-neighbor API](https://docs.ovito.org/python/modules/ovito_data.html#ovito.data.NearestNeighborFinder).

Use neighbor pairs to bin type-resolved distances or construct local Z_ij, Cowley parameters and chemical contrasts; accumulate squared degrees if an exact global random-label variance baseline is needed. Vectorize type lookup and histogramming. Keep float64 for scientific moments unless a precision comparison supports a smaller dtype. Check neighbor-vector dtype: the current `CutoffNeighborFinder.find_all()` signature specifies float32 vectors; casting them to float64 cannot restore lost precision.

## Parallel processing and Slurm

OVITO requires the **spawn** multiprocessing start method on every OS because it creates background threads. Native OVITO objects cannot cross process boundaries: build/reuse each pipeline inside its worker and exchange ordinary paths, scalars or NumPy arrays. Use a standalone script with an `if __name__ == '__main__':` guard; this multiprocessing pattern is not for code executing inside the OVITO Pro GUI. Set `OVITO_THREAD_COUNT` **before importing OVITO**. `ovitos --nthreads N` is another documented thread control. [Official performance guidance](https://docs.ovito.org/python/introduction/scripting/performance.html).

Recommended allocation policy: choose either one process with several OVITO threads or several processes with restricted threads. Keep processes × threads within allocated CPUs, and cap BLAS/OpenMP threads if NumPy/SciPy is also used. Prefer independent composition/replica files or contiguous frame blocks; benchmark compressed-file indexing/decompression and shared-storage traffic before spreading random frames across many readers. This scheduling advice is a workflow recommendation, not a promised OVITO speedup.

For a one-task Slurm allocation, request `--cpus-per-task` and use `SLURM_CPUS_PER_TASK` as the pool budget, not the machine's total CPU count. A Python process pool stays on one node; multiple nodes require separate job-array/tasks orchestration. Ordinary structural analysis should not reserve GPUs merely because MD used GPUs. [Slurm sbatch resource variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES).

Minimal multiprocessing shell (template; `analyze_case` is a top-level worker function that accepts paths and returns compact results):

```python
import os
os.environ["OVITO_THREAD_COUNT"] = "1"  # Before any OVITO import.
import multiprocessing as mp

if __name__ == "__main__":
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for result in pool.imap(analyze_case, case_paths, chunksize=1):
            save_compact_result(result)
```

Before scaling up, compare compressed versus uncompressed input on a small known case, and serial versus parallel outputs on identical frames. Check frame count/timestep, IDs/type mapping, NPT cell volume/PBC, pair reciprocity, coordination sums and RDF normalization. Record wall time, peak memory, worker/thread counts, input sizes and output sizes. Accept parallelization on measured throughput and matching scientific results; never claim speedup from API availability alone.

## Structure identification and rendering

Choose CNA, PTM, centrosymmetry, Voronoi or dislocation analysis according to the observable and the installed API. Verify thresholds against reference structures and thermal/strain effects. Use the modifier's named structure-type constants rather than unexplained integer codes; retain type fractions/counts alongside images. Color legends must distinguish structural labels from chemical species.

For Python rendering, add the evaluated pipeline to the scene with `pipeline.add_to_scene()` before setting the viewport and calling `zoom_all()`/`render_image()`. Confirm the intended frame, simulation cell and visible selection. Record a physically useful viewing direction, legend and time/state. Inspect the produced image for an empty scene, clipping and misleading scale; a rendered snapshot does not replace quantitative structure validation. See [rendering images](https://docs.ovito.org/python/introduction/scripting/rendering.html).
