#!/usr/bin/env python3
"""Read-only UMT checks, offline job rendering and recipe materialization."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))


def ssh_args(c):
    identity = Path(c["identity_file"]).expanduser().resolve()
    # No machine-global SSH configuration, agent forwarding or implicit user identity.
    return ["ssh", "-F", "NUL" if sys.platform == "win32" else "/dev/null",
            "-p", str(c["port"]), "-i", str(identity),
            "-o", 'UserKnownHostsFile="'+Path(c.get("known_hosts_file", ROOT/"known_hosts")).expanduser().as_posix()+'"',
            "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
            "-o", "ForwardAgent=no", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", c["ssh_user"]+"@"+c["host"]]


def doctor(c, remote=False):
    checks = {"profile": True, "ssh_available": bool(shutil.which("ssh")),
              "private_key_exists": Path(c["identity_file"]).expanduser().is_file(),
              "known_hosts_exists": Path(c.get("known_hosts_file", ROOT/"known_hosts")).expanduser().is_file(),
              "software_configured": {k:bool(v) for k,v in c["software"].items()}}
    print(json.dumps(checks, indent=2))
    if remote:
        if not all(checks[k] for k in ("ssh_available", "private_key_exists", "known_hosts_exists")):
            raise ValueError("Configure OpenSSH, your identity file and verified known_hosts before --remote")
        cmd = "\n".join([
            "set -eu", "hostname", "id -un", "printf '\\nPartitions:\\n'",
            "sinfo -o '%P %a %l %D %G'", "printf '\\nYour associations:\\n'",
            'sacctmgr -nP show assoc where user="$(id -un)" format=Cluster,Account,Partition,QOS,MaxWall || true',
            "printf '\\nStorage (login view only; compute write probe is separate):\\n'",
            "ls -ld -- "+" ".join(shlex.quote(c[k]) for k in ["remote_home", "remote_scratch"]),
            "printf '\\nYour queue:\\n'", 'squeue -u "$(id -un)" -o "%.18i %.12P %.32j %.9T %.10M %.6D %R"'
        ])
        subprocess.run(ssh_args(c)+["bash -s"], input=cmd+"\n", text=True, check=True, timeout=90)
    return 0 if all(checks[k] for k in ["ssh_available","private_key_exists","known_hosts_exists"]) else 1


def replace_recipe(c, path, out):
    base=(ROOT/"recipes").resolve()
    src=(base/path).resolve()
    if base not in src.parents or not src.is_file():
        raise ValueError("Recipe must be a file under recipes/")
    dest=Path(out).resolve()
    if dest.exists():
        raise ValueError("Output already exists")
    text=src.read_text(encoding="utf-8")
    values={"@REMOTE_HOME@": c["remote_home"], "@REMOTE_SCRATCH@":c["remote_scratch"],
            "@REMOTE_WORK@":c["remote_work"], "@CLUSTER_USER@":c["ssh_user"],
            "@SLURM_ACCOUNT@":c["slurm_account"],
            "@LOCAL_TIAL@":c["data"].get("local_tial"),
            "@LOCAL_MLIP@":c["data"].get("local_mlip"),
            "@LOCAL_USER@":None, "@LOCAL_MACHINE@":None}
    for key,value in values.items():
        if value is not None:
            if key in ["@LOCAL_TIAL@", "@LOCAL_MLIP@"]:
                value = str(value).replace("\\", "/")
            text=text.replace(key,value)
    unresolved=sorted(set(re.findall(r"@[A-Z_]+@",text)))
    if unresolved:
        raise ValueError("Unresolved template values: "+", ".join(unresolved))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(text,encoding="utf-8",newline="\n")
    print("Materialized reference; review dated run IDs, dependencies and input files before use: "+str(dest))


def render_job(c, a):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}",a.run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}",a.walltime):
        raise ValueError("Use walltime HHH:MM:SS")
    if a.cpus < 1 or a.cpus > 128 or not re.fullmatch(r"\d+[MG]",a.memory):
        raise ValueError("Invalid CPU or memory request")
    if a.backend != "probe" and not a.command:
        raise ValueError("Set --command for this backend")
    dest=Path(a.out).resolve()
    if dest.exists():
        raise ValueError("Choose a new output directory")
    sw=c["software"]
    activation=[]
    if a.backend == "dpa4":
        if not sw["conda_sh"] or not sw["dpa4_env"]:
            raise ValueError("Set software.conda_sh and software.dpa4_env first")
        activation=["source "+shlex.quote(sw["conda_sh"]),"conda activate "+shlex.quote(sw["dpa4_env"])]
    elif a.backend != "probe":
        key={"grace":"grace_env_sh","vasp":"vasp_env_sh",
             "lammps-deepmd":"lammps_deepmd_env_sh","lammps-grace":"lammps_grace_env_sh"}[a.backend]
        if not sw[key]:
            raise ValueError("Set software."+key+" first")
        activation=["source "+shlex.quote(sw[key])]
    partition=c["partitions"][a.resources]
    required = {"partition": partition, "slurm_account": c["slurm_account"]}
    if c.get("qos"):
        required["qos"] = c["qos"]
    if a.resources == "gpu":
        required["gpu_gres"] = c.get("gpu_gres")
    for name, value in required.items():
        if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", value):
            raise ValueError("Set a valid scheduler profile value for " + name)
    remote=c["remote_work"].rstrip("/")+"/runs/"+a.run_id
    if not re.fullmatch(r"/[a-zA-Z0-9_./-]+",remote) or ".." in remote.split("/"):
        raise ValueError("Invalid remote workspace")
    directives=["#!/bin/bash", "#SBATCH --job-name="+a.run_id,
       "#SBATCH --partition="+partition, "#SBATCH --account="+c["slurm_account"],
       "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
       "#SBATCH --cpus-per-task="+str(a.cpus), "#SBATCH --mem="+a.memory,
       "#SBATCH --time="+a.walltime, "#SBATCH --chdir="+remote,
       "#SBATCH --output="+remote+"/slurm-%j.out", "#SBATCH --error="+remote+"/slurm-%j.err"]
    if c.get("qos"):
        directives.append("#SBATCH --qos="+c["qos"])
    if a.resources == "gpu":
        directives.append("#SBATCH --gres="+c["gpu_gres"])
    probe="\n".join(["hostname", "id -un", "pwd", 'test -n "${SLURM_JOB_ID:-}"',
       "touch .compute-write-probe", "rm -- .compute-write-probe", "df -h .",
       "command -v nvidia-smi >/dev/null && nvidia-smi || true"])
    body=["set -euo pipefail", "umask 077", 'test -n "${SLURM_JOB_ID:-}"',
       'mkdir -p cache tmp', 'export XDG_CACHE_HOME="$PWD/cache" TMPDIR="$PWD/tmp"',
       'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1',
       'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
       'export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"']
    text="\n".join(directives+[""]+body+activation+[probe if a.backend=="probe" else a.command])+"\n"
    dest.mkdir(parents=True)
    (dest/"job.sbatch").write_text(text,encoding="utf-8",newline="\n")
    record={"run_id":a.run_id,"state":"prepared_offline","remote_run_dir":remote,
        "backend":a.backend,"resources":a.resources,"cpus":a.cpus,"walltime":a.walltime,
        "script_sha256":hashlib.sha256(text.encode()).hexdigest(),"job_id":None}
    (dest/"submission-intent.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2))
    print("No job submitted. See docs/03-slurm.md for staging, submission and reconciliation.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    d=sub.add_parser("doctor"); d.add_argument("--remote",action="store_true")
    s=sub.add_parser("ssh-command");s.add_argument("--command",required=True)
    r=sub.add_parser("materialize");r.add_argument("--recipe",required=True);r.add_argument("--out",required=True)
    j=sub.add_parser("render-job")
    j.add_argument("--backend",choices=["probe","dpa4","grace","vasp","lammps-deepmd","lammps-grace"],required=True)
    j.add_argument("--resources",choices=["cpu","gpu"],required=True)
    j.add_argument("--run-id",required=True);j.add_argument("--out",required=True)
    j.add_argument("--command");j.add_argument("--cpus",type=int,default=4)
    j.add_argument("--memory",default="8G");j.add_argument("--walltime",default="00:05:00")
    a=p.parse_args();c=config()
    if a.action=="doctor":return doctor(c,a.remote)
    if a.action=="ssh-command":
        print("Executing the explicitly supplied remote shell command.",file=sys.stderr)
        return subprocess.run(ssh_args(c)+[a.command],timeout=120).returncode
    if a.action=="materialize":replace_recipe(c,a.recipe,a.out)
    if a.action=="render-job":render_job(c,a)
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(2)
