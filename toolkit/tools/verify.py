#!/usr/bin/env python3
"""Verify archive contents against SHA256SUMS.json without third-party packages."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/"SHA256SUMS.json").read_text(encoding="utf-8"))
errors=[]
for name,want in manifest.items():
    p=(root/name).resolve()
    if root not in p.parents:
        errors.append("invalid manifest path: "+name);continue
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=want:
        errors.append(name)
actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        and p.name!="SHA256SUMS.json" and "__pycache__" not in p.parts}
errors += ["unlisted: "+x for x in sorted(actual-set(manifest))]
print(json.dumps({"checked":len(manifest),"errors":errors},indent=2))
sys.exit(bool(errors))
