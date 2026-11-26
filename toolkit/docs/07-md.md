# MD: ASE NPT и LAMMPS NPT/NVT

**Текущая политика DPA4.** До повторных оценок, фононов и MD заморозьте исходный checkpoint командой `dp --pt freeze -c CHECKPOINT.pt -o MODEL.pt2` с `DP_TRITON_INFER=2` во время экспорта. Для эталонного режима float32 задайте `DP_TF32_INFER=0` и `DP_AMP_INFER=0`. Зафиксируйте версии, хеши и допуски. Сверьте энергию, силы и напряжения на тех же структурах и загружайте принятый `.pt2` в основном расчёте. После ошибки экспорта или сравнения проведите диагностику. Исторические примеры eager/Triton 0 относятся к прежним сериям. Native LAMMPS требует собственной проверки.

Переносится практический опыт TiAl: составные ряды жидкостей, density/enthalpy NPT,
независимые transport replicas и проверка checkpoint/type_map/export.
Исходные thermodynamic protocols сохранены как .source и нуждаются в новых путях/весах/данных.

ASE NPT: `TiAl/audit/oct29/xc_liquid_density_extended/md_common.py.source`.
Native DeepPot adapter возвращает E, F и stress; tensor симметризуется и делится на V
с минусом. Перед MelchionnaNPT ячейка приводится к upper triangular с согласованным
поворотом координат. `set_fraction_traceless(0)` задаёт выбранную изотропную настройку;
параметр barostat связан с выбранным bulk modulus. Это часть протокола, не константа для всех сплавов.
Термостат/баростат, dt, sampling, equilibration и seeds фиксируются в protocol.json.

LAMMPS production: `TiAl/audit/oct29/pbe15k_ti_temperature/md_core.py.source`.
Перед серией сверяются hashes checkpoint/frozen model, полный type_map и числа атомов.
Binary chemical system не означает binary type_map у foundation checkpoint.
Native DeepPot и LAMMPS run0 сравниваются по E/F/pressure; затем bench100 steps и
оценка времени серии. Historical tolerances здесь — пример, их обоснование для новой задачи
фиксируется до просмотра ошибок. Фактическая A100/DEVICE и число GPU подтверждаются runtime.

Для плотности и энтальпии — подходящий NPT production после equilibration.
Для Green–Kubo вязкости — NVT production при выбранном volume после NPT reservoir,
с сохранением pressure tensor на каждом требуемом шаге. Сохранение только SACF
не позволяет восстановить сырые pressure time series или полноценную STACIE статистику.
Не склеивайте конец одной реплики с началом другой. Разные seeds ещё не гарантируют
независимость при общем reservoir; фиксируйте родительские конфигурации и промежутки времени.

`stage()` в одном из исходных ASE рецептов переинициализирует velocities.
Продление от последнего frame без полного состояния integrator/thermostat является новым
этапом с оговорённой re-equilibration, а не точным restart.
Сохраняйте trajectory, thermo, final frame, step/time, seed, model/protocol/code hashes,
номер Slurm job и фактические elapsed/resources. Новые attempts пишутся в новый каталог.
После окончания проверяются NaN, температура/давление, volume/density, фаза, drift и
продолжительность статистики. Успешный run0 не заменяет physical/stability validation.
