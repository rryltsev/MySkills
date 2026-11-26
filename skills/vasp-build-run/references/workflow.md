# VASP: сборка, PAW, расчёты

Исходный рабочий CPU/MPI build: VASP 6.6.1, oneAPI compiler/MKL 2025.3 и MPI 2021.17.
Прямые shared scripts под `/misc/umt_shared/opt/intel/oneapi/...` в
`recipes/MLIP-research/scripts/hpc/vasp-env.sh.source` обходили различие Lmod на login/compute.
Сначала подтвердите наличие этих общих версий. Путь пользователя к VASP задаётся в новом профиле.

Сборка: новый private prefix на writable storage; разрешённый licensed source archive;
upstream arch/makefile.include.oneapi + сохранённые oneapi-overrides.mk; `mpiifx`,
sequential MKL и VASP_TARGET_CPU=-xSSE4.2. В исходном build применялся `-fp-model precise`
для vdw_nl.o как локально зафиксированная коррекция. Её применимость к другой версии
проверьте по первичному upstream/инструкции к выбранному build, не расширяйте на все файлы.
`make DEPS=1 -j N std/gam/ncl` выполняется в Slurm; затем ldd и бинарные SHA-256.

Исторический ограниченный тест охватывал четыре случая: bulk_AlPAW, H2Ostretch_efor,
Fe_r2SCAN_ncl_soc, bulk_BN_vdW-DF3-opt1. Это не полный upstream testsuite.
GPU/HDF5/дополнительные интерфейсы этим build не квалифицированы. Сохраняйте build log,
toolchain identity, exact sources и все результаты проверок. Пример запуска внутри allocation:
`source /YOUR/vasp/env.sh`, затем `mpirun -np "$SLURM_NTASKS" vasp_std`.
Для srun согласуйте PMI/MPI с кластером; пример phonon workflow использовал srun отдельно.

PAW: библиотека 64, PBE или LDA, точный вариант по каждому элементу в порядке POSCAR.
Hash-checking сборщик — `scripts/vasp_potentials.py.source`; установка реестра —
`scripts/install_paw.py.source`. В новый расчёт сохраняют potcar-provenance.json.
Нельзя незаметно смешивать семейства/варианты, менять Ti/Ti_pv/Ti_sv или заменять
production potentials тестовыми. Для PBE/PBEsol/r2SCAN согласуйте PAW и электронные настройки.

Отдельно фиксируйте energy/free_energy convention, smearing, magnetism, k-grid, ENCUT,
EDIFF/NELM, forces/stress units и знаки. Валидируйте OUTCAR/vasprun.xml, normal termination,
достижение electronic convergence, ionic convergence для relax и finite labels.
Косвенный критерий числа SCF iterations дополняйте реальным статусом сходимости.

ZIP содержит только собственные build/validation/PAW assembly scripts и описания.
Licensed VASP source/binaries, testsuite POTCAR и PAW contents новый пользователь получает отдельно.
Рецепты: `scripts/install_vasp.py.source`, `hpc/build-vasp.sbatch.source`,
`hpc/vasp-env.sh.source`, `configs/vasp/oneapi-overrides.mk.source`.
