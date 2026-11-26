# Независимые DFT holdouts и фононы

**Текущая политика DPA4.** До повторных оценок, фононов и MD заморозьте исходный checkpoint командой `dp --pt freeze -c CHECKPOINT.pt -o MODEL.pt2` с `DP_TRITON_INFER=2` во время экспорта. Для эталонного режима float32 задайте `DP_TF32_INFER=0` и `DP_AMP_INFER=0`. Зафиксируйте версии, хеши и допуски. Сверьте энергию, силы и напряжения на тех же структурах и загружайте принятый `.pt2` в основном расчёте. После ошибки экспорта или сравнения проведите диагностику. Исторические примеры eager/Triton 0 относятся к прежним сериям. Native LAMMPS требует собственной проверки.

TiAl testset использовал geometry-only diversity по periodic SOAP, balanced family/composition
quotas, parent trajectory separation и проверку совпадений с train/reference.
Конкретные Ti-Al species/quotas/порог1.8Å — historical recipe; для новой химии они меняются.
Близкие структуры ищут с учётом PBC, species assignment и деформации ячейки.
Сохраняйте SHA-256 membership и selection protocol до расчёта новых DFT labels.
Нельзя выбирать final test по prediction errors или использовать его для checkpoint tuning.

Рецепты: `TiAl/testset/{audit_sources,select_testset,finalize_testset,verify_testset}.py.source`.
В source присутствует pickle чтение исходного project cache: используйте только собственные
доверенные cache files либо замените формат для внешних данных.
Сами dataset/trajectories/labels в ZIP не включены. requirements-lock сохранён, но
не заменяет проверку Linux environment и совместимости выбранных descriptors.

DFT: `testset/dft/test50-dft-20261021-01/workflow.py.source` и protocol.json.source
показывают bounded PBE/PBEsol/r2SCAN workflow. Сверяйте normal termination, SCF convergence,
finite E/F/stress и одинаковую геометрию для single-point сравнения. Успех Slurm не критерий DFT.

Phonopy/VASP: `TiAl/phonons/scripts/workflow.py.source`, `production_step_v10.py.source`,
`checkpoint_guard_v5.py.source`, `build_pbesol_phonons.py.source`.
Фиксируются XC, PAW variant/hash, ENCUT, kmesh, smearing, relax criteria, masses,
supercell, displacements ±δ, finite-displacement symmetry, acoustic-sum treatment,
qmesh, broadening и DOS normalization. Контролируются residual stress/forces и force drift.
Проверяйте convergence по supercell/displacement/kmesh, а не выбирайте красивый DOS.

Отдельно сравниваются: DFT vs MLIP при одной DFT-equilibrium geometry и predictions
каждого метода при собственной equilibrium volume. Иначе volume effect смешивается
с ошибкой force constants. Model checkpoint определяется hash и internal step metadata,
а не только надписью5k/15k. Imaginary modes не удаляются; проверяются physics/numerics.

Рецепты matched pairs: `dpa4_matched_forces.py.source`, `analyze_dpa4_pair.py.source`,
`verify_dpa4_pairs.py.source`, `compare_xc_checkpoint_phonons.py.source`.
Выбранные launch/recovery snapshots могут требовать других versioned helpers и manifests;
они не представляют автономную готовую phonon pipeline и не исполняются автоматически.
