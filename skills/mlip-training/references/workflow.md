# Обучение, fine-tuning, LoRA и restart

В MLIP-research есть synthetic GPU smoke, fine-tuning исходных FM, restart Adam,
оценка исходных моделей на DFT, реальное Al32 fine-tuning и LoRA trial.
Все эти этапы имеют разные критерии успеха. Не использовать synthetic checkpoints
как научный teacher и не выдавать наличие элемента в type_map за точность для этого элемента.

Рецепт реального Al32 пилота: 21 train / 6 validation / 13 test, seed42, batch2,
Adam 1e-4 → 1e-5; DPA4 1000 updates, GRACE 100 epochs. Это исторические параметры,
а не универсальные настройки под TiAl или новую задачу. Точные configs сохранены как .source.
Grouped split делается по parent structure/trajectory/system; тесно связанные кадры не
пересекают train/test. Сохраняются stable configuration IDs, input hashes, source checkpoint
и training versions. Test labels не входят в training/selection payload.

Energy-reference diagnostic: delta=mean_train((E_DFT-E_model)/N),
E_aligned=E_model+N*delta. Report raw+aligned; силы и stress не меняются. Константа не
исправляет зависимую от состояния ошибку и не доказывает эквивалентность PAW.
Для многокомпонентной системы правило reference correction определяется отдельно.

Validation selector заранее фиксирует metric/scales и сравнивает исходную FM с обученными
кандидатами. Исходная FM тоже может победить. После выбора checkpoint hash и candidate ID
блокируются, test оценивается один раз для финального отчёта. Улучшение train loss или
части test metrics само по себе не доказывает улучшение MD/фаз/свойств.

Restart переносит веса, optimizer state и schedule, а не только network weights.
В проверенной DeePMD 3.2.0 конфигурации после Adam count20 native loop index был19.
`hpc/restart-child.py.source` корректировал следующий loop start до20 и LR horizon.
Это узкая version/setup-specific поправка: не применять к другим версиям без проверки.
Побитная эквивалентность uninterrupted trajectory обучения не была доказана.
GRACE restart также проверялся на сохранении optimizer state и продолжении updates.

DPA4 LoRA trial и код сохранены в `hpc/al32-lora-run.py.source`, `hpc/lora-child.py.source`,
`scripts/assess_lora.py.source`, `configs/training/al32-dpa4-lora-v1.json.source`.
Наличие реализации LoRA не объявляется доказанным преимуществом на новом dataset.

Основные рецепты: `hpc/fm-finetune.py.source`, `hpc/fm-child.py.source`,
`hpc/al32-finetune-run.py.source`, `scripts/finetune_selection.py.source`,
`scripts/assess_finetune.py.source`, `scripts/diagnose_dpa4.py.source`.
Для исходных весов сохранены metadata, URLs/revisions/hashes/licences в `models/foundation/*.json.source`.
Сами веса загружают/передают отдельно с проверкой разрешения и SHA-256.
