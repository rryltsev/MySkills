# DFT → foundation teacher → distilled student

Навык mlip-distillation извлечён из MLIP-research. Это методическая схема;
готовая научно квалифицированная end-to-end дистилляция данным архивом не заявляется.

1. Определите chemistry/domain/observables: bulk liquid/solid, interfaces, surface terminations,
   segregation, strain и physically relevant short separations.
2. Введите stable configuration/parent IDs. Храните raw DFT или durable artifact references,
   settings/PAW hashes, magnetism, smearing, convergence и units/stress conventions.
3. Заморозьте grouped train/validation/test; отдельно храните fine-tuning и replay data.
   Изучите возможное перекрытие с FM pretraining, насколько позволяет provenance.
4. Сравните starting FM и adapted teacher с независимым DFT и нужными physical observables.
   Определите допустимые ошибки под конкретный observable и область применения.
5. Зафиксируйте accepted teacher checkpoint/hash. Teacher predictions — pseudo-labels,
   не DFT truth: для каждой метки нужны source, conditions и weights при mixing с DFT.
6. Выберите student по реальным throughput и accuracy на целевом hardware/system size.
7. Сравните student с teacher и независимым DFT, затем deployed/training equivalence,
   MD stability, phase energetics и нужные liquid/interface observables.
8. Active learning запускается с explicit acquisition criteria и DFT budget; frozen test
   не используется для выбора. Negative evidence и limits публикуются вместе с моделью.

Архитектуры остаются расширяемыми: DPA4, DPA4c, GRACE, legacy DeepMD student — варианты.
`configs/model_backends.json.source` описывает будущий adapter contract; универсальный
рабочий adapter в исходном проекте ещё не реализован, и этот набор его не обещает.
Исходник: `recipes/MLIP-research/.agents/skills/mlip-distillation/SKILL.md.source`.
