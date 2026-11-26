# Slurm: подготовка, запуск и восстановление

Из нового проекта подготовьте probe. Этот шаг полностью локальный и не ставит задания в очередь:

```text
python .mlip/tools/cluster.py render-job --backend probe --resources cpu --run-id setup-probe-01 --out work/setup-probe-01 --cpus 1 --memory 1G --walltime 00:02:00
```

Генерируются job.sbatch и submission-intent.json с remote_run_dir и SHA-256.
Перед отправкой проверьте реальные partition/account/QOS, лимиты и разрешённый бюджет.
На сервере создайте НОВЫЙ remote_run_dir, перенесите файлы через scp/SFTP с теми же
identity_file/known_hosts и StrictHostKeyChecking=yes. Каталог для sbatch stdout/stderr
должен существовать до submit и быть доступен на compute-узле: ошибка открытия stdout
может произойти до старта bash и не исправляется mkdir внутри job.

Для scp Windows/Linux используйте параметры личного ключа и host-key базы как в SSH;
порт задаётся `-P 22` (у ssh `-p`). Путь known_hosts делайте абсолютным при смене cwd.
Пример запуска в SSH из удалённого каталога: `sbatch --parsable job.sbatch`.
Запишите job ID немедленно в отдельный submission.json вместе с UTC, remote_run_dir,
hash script/input и явным resource request. Не редактируйте принятую копию входов.

```text
squeue -j JOB_ID -o "%.18i %.9T %.10M %R"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES -P
```

Если SSH оборвался во время sbatch, состояние UNKNOWN: сначала найдите job по run_id,
времени, пользователю и рабочему каталогу через squeue/sacct/scontrol, и лишь затем решайте
вопрос о повторе. При неясности не дублируйте submission. Старые job IDs в рецептах
не являются задачами нового пользователя и не должны передаваться scancel/scontrol.

Probe проверяет запись только в собственном run directory; это не проверка прав на все
будущие env/software/results. Для GPU probe задайте `--resources gpu` и сверяйте модель
фактического GPU. Подготовка шаблона не расходует кластерный бюджет; запуск расходует.

Для пользовательской команды, после настройки software paths:

```text
python .mlip/tools/cluster.py render-job --backend dpa4 --resources gpu --run-id my-train-01 --out work/my-train-01 --cpus 4 --memory 16G --walltime 01:00:00 --command "dp --pt train input.json"
```

Эта команда формирует wrapper, но не копирует input.json, datasets или checkpoints.
Добавьте полный payload и input-hashes до отправки. Генератор предназначен для одного узла,
одной задачи и при GPU одного GRES; многопроцессный VASP/MPI требует отдельного осознанного
ресурсного скрипта. Для CPU MPI пример — historical `build-vasp.sbatch.source`.
Walltime и CPU/GPU-hours подтверждаются под конкретный запуск: в профиле budget по умолчанию null.

COMPLETED не равен принятому научному результату: проверьте ExitCode, полноту, finite values,
сходимость и ожидаемое число шагов/кадров. Отдельно сохраняйте failed/cancelled attempts.
Копируйте выходы без destructive sync, сверяйте SHA-256 на обеих сторонах; manifest сам
по себе не резервная копия. Держите фактическую долговременную копию вне временного scratch.

Рецепты: `scripts/run_mlip_smoke.py.source`, `scripts/finetune_cluster.py.source`,
`scripts/deployment_cluster.py.source` в `recipes/MLIP-research/`.
