#!/usr/bin/env python3
"""Prepare a local .mlip profile and reference resources. Uses the Python standard library."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

TOOLKIT = Path(__file__).resolve().parents[1]


def value_or_prompt(value, label):
    if value:
        return value
    answer = input(label + ": ").strip()
    if not answer:
        raise ValueError(f"{label} is required")
    return answer


def identifier(value, label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {label}")
    return value


def remote_path(value):
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in value.split("/") or not value.strip("/"):
        raise ValueError("Specify a non-root absolute remote path without spaces or '..'")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project")
    parser.add_argument("--user", help="UMT SSH login chosen for this profile")
    parser.add_argument("--account", help="Slurm accounting association for this login")
    parser.add_argument("--remote-home")
    parser.add_argument("--remote-scratch")
    parser.add_argument("--identity-file", help="Path to the user's own SSH key")
    parser.add_argument("--known-hosts-file", help="Path to a locally maintained, verified OpenSSH host-key file")
    parser.add_argument("--host", default="umt.imm.uran.ru")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--resources-only", action="store_true", help="Compatibility option; the initializer always installs only the project profile/resources")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()
    if args.non_interactive and not all((args.project, args.user, args.account, args.remote_home,
                                        args.remote_scratch, args.identity_file)):
        parser.error("Specify project, user, account, remote-home, remote-scratch and identity-file")
    if not 0 < args.port <= 65535:
        parser.error("Port must be between 1 and 65535")

    project = Path(value_or_prompt(args.project, "Local project directory")).expanduser().resolve()
    if project == TOOLKIT or TOOLKIT in project.parents or project in TOOLKIT.parents:
        raise ValueError("Choose a project directory separate from the toolkit")
    target = project / ".mlip"
    if target.exists() or target.is_symlink():
        raise ValueError(f"Existing project profile at {target}")
    user = identifier(value_or_prompt(args.user, "SSH login"), "login")
    account = identifier(value_or_prompt(args.account, "Slurm account"), "account")
    scratch = remote_path(value_or_prompt(args.remote_scratch, "Compute-writable remote directory"))
    remote_home = remote_path(value_or_prompt(args.remote_home, "Remote home"))
    key = Path(value_or_prompt(args.identity_file, "Local SSH identity file")).expanduser().resolve()
    known = Path(args.known_hosts_file).expanduser().resolve() if args.known_hosts_file else target / "known_hosts"
    profile = {
        "schema_version": 2, "host": identifier(args.host, "hostname"), "port": args.port,
        "ssh_user": user, "slurm_account": account, "identity_file": str(key), "known_hosts_file": str(known),
        "remote_home": remote_home, "remote_scratch": scratch, "remote_work": scratch + "/mlip-workspace",
        "partitions": {"cpu": None, "gpu": None}, "qos": None, "gpu_gres": None,
        "software": dict.fromkeys(("conda_sh", "dpa4_env", "grace_env_sh", "vasp_env_sh",
                                   "lammps_deepmd_env_sh", "lammps_grace_env_sh", "paw_root")),
        "allocation_budget": {"cpu_hours": None, "gpu_hours": None},
        "data": {"local_tial": None, "local_mlip": None},
    }
    project.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mlip-setup-", dir=project) as staging:
        pending = Path(staging) / "profile"
        pending.mkdir()
        for folder in ("docs", "recipes", "environments", "templates"):
            shutil.copytree(TOOLKIT / folder, pending / folder)
        (pending / "tools").mkdir()
        for name in ("cluster.py", "capture_environment.py"):
            shutil.copyfile(TOOLKIT / "tools" / name, pending / "tools" / name)
        shutil.copyfile(TOOLKIT / "provenance.json", pending / "provenance.json")
        (pending / "config.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        (pending / ".gitignore").write_text("*\n", encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise ValueError(f"Existing project profile at {target}")
        pending.rename(target)
    print(f"Prepared {target}. Review the cluster settings, then run the local doctor.")
    print(f'python "{target / "tools" / "cluster.py"}" doctor')


if __name__ == "__main__":
    try:
        main()
    except (EOFError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
