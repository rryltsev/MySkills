# Плотность, энтальпия, диффузия, вязкость

Density: rho[g/cm3]=sum(mass[amu])/V[Å³] × 1.66053906660.
Оценка ошибки должна учитывать временную корреляцию и независимые реплики.
Сопоставление рядов требует одинаковых T, external P, phase и model hash.

ASE stress положителен при растяжении: P=-mean(sigma_xx,yy,zz).
Для полного pressure в MD учитывайте kinetic contribution (`include_ideal_gas=True`).
Термодинамическая энтальпия при внешнем давлении:
H_ext=E_pot+E_kin+P_external*V. Не подставляйте -P*V из-за неверного знака stress.
Если отдельно изучается instantaneous-pressure estimator, явно обозначьте это определение.
DeltaH_mix(x)=H_alloy-x*H_Al-(1-x)*H_Ti; endpoints должны иметь те же T/P/model/protocol.
Сохраняйте исходный legacy estimator для аудита; пересчёт величины не должен маскироваться
под новые MD данные. Константа eV/atom→kJ/mol(atom): примерно96.4853321233.

Diffusion/MSD: корректно unwrap PBC, учитывайте изменение ячейки и COM drift,
выбирайте диффузионное временное окно и контролируйте фазу. Один финальный MSD —
диагностика мобильности, а не автоматически достоверный коэффициент диффузии.

## Вязкость / STACIE

Храните сырые pxx pyy pzz pxy pxz pyz по каждой реплике, step/time, dt, V, T и units.
В units metal pressure bar→Pa; rho/enthalpy используют другие единицы.
Из pressure строятся согласованно нормированные ортогональные сдвиговые каналы.
Смотрите точную нормировку в сохранённом analysis-коде; пять каналов не равны пяти
независимым репликам. Green–Kubo prefactor V/(kB*T) берётся для каждой реплики.
Спектры усредняются, временные ряды реплик не склеиваются.

В историческом аудите использовались STACIE LorentzModel, CV2LCriterion,
маргинализация cutoff, включая zero frequency. Параметры и версии исходников фиксируются.
Оценивайте cutoff_Z/cost_Z, neff, half-window sensitivity, full-replica jackknife,
подготовку ансамбля, thermostat и длину trajectories. Увеличение neff принудительным
расширением плохого spectral window не доказывает улучшение точности.
Не подбирайте cutoff под гладкость eta(T). Fit Arrhenius — отдельная модель тренда;
он не заменяет исходные оценки и не уменьшает их ошибки задним числом.

Проверяйте duplicate raw files по hashes; копии данных не новые реплики.
Сверяйте реальные temperature/thermostat logs: в TiAl1941 K и1943 K оказались разными сериями.
Незначимое различие checkpoints не доказывает их эквивалентность; отсутствие model hash
в экспорте ограничивает provenance. Не смешивайте старые/обновлённые series без overlap audit.

Рецепты: `TiAl/audit/oct20/reprocess_thermo.py.source`, `pressure_audit.py.source`,
`TiAl/audit/nov01/r2scan_viscosity_diagnostics/{analyze,finalize,supplement,windows}.py.source`,
`TiAl/audit/oct27/viscosity_5k_vs_15k/analyze.py.source`.
STACIE сторонние исходники и зависимости не переупакованы; stacie_source_hashes.json.source
помогает сверить версию, а package/API и missing raw inputs надо восстановить отдельно.
