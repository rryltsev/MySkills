# DPA4 / DPA4c / GRACE: окружения и установка

Это зафиксированные рабочие сочетания из MLIP-research, главным образом от 17–21 октября 2026.
Новая установка на другом профиле ещё не проверена. Сначала определите доступность общего
окружения; не предполагается право чтения или модификации старого пользовательского prefix.

## DeePMD

На A100 проверены Python 3.12, DeePMD-kit 3.2.0, Torch 2.11.0+cu130, vesin-torch 0.6.1
в `lammps_deepmd_dpa4`. Команды: DPA4 `dp --pt train input.json`,
DPA4c `dp --pt-expt train input.json`. Альтернативное окружение `deepmd_320_a100`
проходило DPA4c, но DPA4 там не обучался без Torch neighbor-list dependency.
Для K40m подтверждён только старый se_e2_a: DeepMD 2.2.10, TF 2.9, CUDA 10.2;
из этого нельзя выводить работоспособность DPA4 или современного GRACE.

Полного DPA4 lock-файла нет. Если администратор разрешит доступ к совместимому source prefix,
создайте собственный conda clone или сделайте `conda list --explicit` и `pip freeze --all`
через `tools/capture_environment.py` в Slurm. Такой freeze ещё не фиксирует хеши всех wheel.
Для новой сборки используйте отдельный prefix, Python 3.12 и указанные версии как начальный
рецепт. В `environments/dpa4-a100-observed-freeze.txt` сохранён полный pip inventory
успешного A100 smoke; freeze выводит torch2.11.0 без local CUDA suffix, поэтому cu130
дополнительно зафиксирован выше. `templates/install-dpa4.sh.template` содержит стартовую
установку core packages с этими constraints; это новый рецепт, не проверенная чистая установка.
Подберите wheel Torch по драйверу/GPU согласно официальной инструкции. Не устанавливайте
их поверх GRACE. После установки обязательны pip check и малый GPU train smoke, а затем
проверка экспорта. Установка здесь не представлена как гарантированно побитно воспроизводимая.

Официальные первичные источники для проверки команды/изменений версии:
https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html
https://github.com/deepmodeling/deepmd-kit
Современная документация может описывать новый graph export; исторический edge_vec export
нельзя автоматически заменить без сравнения предсказаний.

## GRACE

Linux x86_64: Python 3.11.17, TensorPotential 0.6.1, TensorFlow 2.20.0,
tf-keras 2.20.1. Сохранены точный conda explicit и 82 wheel с SHA-256.
Восстановление в НОВЫЙ prefix внутри Slurm:

```bash
conda create --yes --prefix "$NEW_PREFIX" --file grace-a100-conda-linux-64.lock.txt
"$NEW_PREFIX/bin/python" -m pip install --no-deps --require-hashes -r grace-a100-pip-linux-64.lock.txt
"$NEW_PREFIX/bin/python" -m pip check
```

Подготовьте prefix и файлы заранее в compute-writable каталоге. Ссылки на пакеты требуют
доступа к сети либо предварительно загруженного wheel/conda cache; ZIP не содержит самих пакетов.
Файл `grace-a100.lock.json` содержит hashes locks и сведения о происхождении.
`proposed.yml` — первоначальное предложение, а не замена принятому lock.

Используйте шаблон `templates/activate-grace.sh.template`, вписав свои пути. Он задаёт
TF_USE_LEGACY_KERAS=1, TF_FORCE_GPU_ALLOW_GROWTH, путь к packaged cuda_nvcc, ptxas/libdevice
и XLA_FLAGS. Рабочая команда: `gracemaker input.yaml`.
Сначала подтвердите import, видимость GPU, synthetic training с finite loss и checkpoint,
затем фактическое fine-tuning. Проверка пяти эпох на гармонических парах — тест ПО.

## Установки на compute-узле

Login-узел: лёгкая подготовка и передача. Сборки, установка окружений и GPU-тесты — Slurm.
На compute-узле одного CONDA_PKGS_DIRS недостаточно: плагины conda пишут в HOME.
В отдельном subshell установщика назначьте writable HOME, XDG_CACHE_HOME, TMPDIR,
CONDA_PKGS_DIRS и PIP_CACHE_DIR. Это локальная настройка дочернего процесса, а не изменение
пользовательского home в системе или переменных оболочки Codex.
Сохраняйте pip report, freeze, версии, хеши и не модифицируйте действующий prefix.

Рецепты: `MLIP-research/scripts/hpc/install-grace.sbatch.source`,
`grace-env.sh.source`, `mlip-smoke.py.source`, `grace-smoke.py.source`,
`grace-smoke.sbatch.source`, `scripts/lock_grace.py.source`.
Официальный проект: https://github.com/ICAMS/grace-tensorpotential
