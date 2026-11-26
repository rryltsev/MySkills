#!/usr/bin/env python3
"""Capture the active Linux environment within a Slurm allocation; no installation."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run environment inspection inside an allocated Slurm compute job")
out=Path(sys.argv[1] if len(sys.argv)>1 else "environment-snapshot")
out.mkdir(exist_ok=False)
record={"python":sys.version,"prefix":sys.prefix,"job_id":os.environ["SLURM_JOB_ID"],
        "packages":{d.metadata["Name"]:d.version for d in md.distributions()},"commands":{}}
commands={"pip-freeze.txt":[sys.executable,"-m","pip","freeze","--all"],
          "pip-check.txt":[sys.executable,"-m","pip","check"],
          "gpu.txt":["nvidia-smi"]}
if shutil.which("conda"):
    commands["conda-explicit.txt"]=["conda","list","--prefix",sys.prefix,"--explicit"]
for filename,cmd in commands.items():
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        (out/filename).write_text(r.stdout+r.stderr,encoding="utf-8")
        record["commands"][filename]={"argv":cmd,"returncode":r.returncode}
    except (OSError,subprocess.SubprocessError) as exc:
        record["commands"][filename]={"argv":cmd,"error":str(exc)}
record["file_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
(out/"manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(out)
