# Исследовательский дизайн и передача состояния

Переработанный исходный mlip-study применим к heterogeneous nucleation / interfacial liquid order.
Для новой задачи создайте собственный brief и state. Исходные научные приоритеты не являются
автоматически целью нового пользователя, не переносите их в глобальные инструкции.

Историческая causal hypothesis: surface chemistry/structure → segregation → liquid order →
precursors → nucleation barrier/phase selection. Проверяйте альтернативные объяснения:
lattice mismatch, strain, surface termination, roughness, wetting и temperature.
Корреляция descriptor с outcome ещё не доказывает mediation/causality.

Отбор системы учитывает chemical coverage, relevant competing phases, experiment,
DFT tractability и проверенную литературу. Al alloys с TiB2/TiC/AlB2/Al3Ti — кандидаты,
а не принятое решение. Отсутствие результата поиска не доказывает novelty.
Определяйте intervention/comparison/falsification и отложенный prospective prediction.

Nucleation rate, induction time и free-energy barrier — разные величины. Учитывайте
censored no-event trajectories и competing phases; enhanced sampling требует проверки CV
и convergence. MD undercooling и timescale не переносятся на эксперимент без обоснования.

Состояние нового проекта храните в человекочитаемом brief и machine-readable state:
цель, решения, open questions, software/model hashes, dataset versions, run_id/job_id,
remote path, статус/failed attempts, actual artifact copies и следующий проверяемый шаг.
Каждое число/figure связано с входами, кодом, моделью и analysis settings.
Historical budget, доступы и прежние разрешения пользователя не наследуются новым проектом.

Исходник: `recipes/MLIP-research/.agents/skills/mlip-study/SKILL.md.source`.
