# Как пользоваться исходными рецептами

Папка recipes хранит выбранные научные и инфраструктурные исходники MLIP-research и TiAl в переносимых текстовых снимках. В MLIP-research представлены модели обучения, схемы валидации, Slurm-рецепты, сборка и развёртывание. В TiAl сохранены подготовка независимых структур, DFT, фононы, анализ жидкостей и диагностика вязкости. Полные учебные навыки, включая более поздние работы, собраны в skills в корне коллекции.

В .source заменены старые home/work/login/account и локальные пути на @TOKENS@.
Метки исторических запусков и хеши иллюстрируют происхождение данных. Для нового проекта задайте новые идентификаторы.
В provenance.json сохранены доступные хеши оригиналов, значения из предыдущего архива и SHA-256 переносимой копии.
Хеши входов внутри historical manifests относятся к старым данным; они могут перестать
совпадать после templating. Для нового расчёта пересоздайте protocol и input manifest.

Материализация одного рецепта из установленного проекта:

```text
python .mlip/tools/cluster.py materialize --recipe MLIP-research/scripts/hpc/grace-env.sh.source --out work/activate-grace-reference.sh
```

Команда подставляет профиль, не исполняет скрипт и не перезаписывает output.
Данные @LOCAL_TIAL@/@LOCAL_MLIP@ задаются в profile.data. Не все произвольные локальные
пути проекта можно автоматически угадать: остаточные placeholders вызывают отказ.
После подстановки проверьте все file dependencies, old run directories, conda env names,
license/access, CUDA layout, resource requests, absolute paths и output destinations.
Templates в templates/ заполняются отдельно; profile software.* — авторитетный список
действительных активаций. Не используйте автоматически путь, полученный простой заменой login.

Быстрый поиск нужного кода: `rg --files .mlip/recipes`; затем ищите функцию/символ
по конкретному файлу. Чтобы изучить unit/sign handling, читайте local MD/analysis source;
чтобы повторить старый результат полностью, запросите его собственный data/model archive
с разрешённым доступом и контрольными суммами.

Исключены: приватные и публичные пользовательские SSH-ключи, пароли/токены,
Codex auth/config/sessions, VASP licensed source/binaries/POTCAR, weights,
trajectories/datasets/raw outputs, personal histories, venvs и third-party vendored libraries.
Доверенный ключ сервера и SSH-профиль настраиваются локально.
