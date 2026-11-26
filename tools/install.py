"""Install the collection's Codex skills from a persistent local checkout (Python 3.9+)."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POINTER = "collection-root.txt"
RECEIPT = ".myskills-install.json"
DEFAULT_DEST = Path.home() / ".agents" / "skills"


def ignore_cache(directory, names):
    return {n for n in names if n == "__pycache__" or n.endswith(".pyc")}


def inventory(folder):
    result = {}
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic link inside skill: {path}")
        if path.is_file() and relative.as_posix() not in (POINTER, RECEIPT):
            result[relative] = hashlib.sha256(path.read_bytes()).digest()
    return result


def classify(source, target):
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        return "conflict"
    if not target.exists():
        return "missing"
    pointer = target / POINTER
    receipt = target / RECEIPT
    if any(p.is_symlink() or not p.is_file() for p in (pointer, receipt)):
        return "conflict"
    try:
        owner = json.loads(receipt.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return "conflict"
    if owner != ownership_record(source):
        return "conflict"
    if pointer.read_text(encoding="utf-8").strip() == str(ROOT) and inventory(target) == inventory(source):
        return "current"
    return "changed"


def ownership_record(source):
    return {"collection": "rryltsev/MySkills", "format": 1, "skill": source.name}


def install_one(source, target, status, backups):
    with tempfile.TemporaryDirectory(prefix=".myskills-", dir=target.parent) as stage:
        prepared = Path(stage) / source.name
        shutil.copytree(source, prepared, ignore=ignore_cache)
        (prepared / POINTER).write_text(str(ROOT) + "\n", encoding="utf-8")
        (prepared / RECEIPT).write_text(json.dumps(ownership_record(source)) + "\n", encoding="utf-8")
        backup = None
        if classify(source, target) != status:
            raise ValueError(f"Installation changed during copy. Retry: {target}")
        if status == "changed":
            backups.mkdir(parents=True, exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix=source.name + "-", dir=backups))
            backup = saved / source.name
            target.rename(backup)
        try:
            prepared.rename(target)
        except OSError:
            if backup and not target.exists():
                backup.rename(target)
            raise
        print(f"Installed {target}" + (f" (previous copy: {backup})" if backup else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skills", nargs="*", help="skill names; default: all")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target skills directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="show available skill names")
    mode.add_argument("--check", action="store_true", help="check the selected installed copies without writing")
    mode.add_argument("--update", action="store_true", help="back up and refresh previously managed copies")
    args = parser.parse_args()
    sources = {file.parent.name: file.parent for file in sorted((ROOT / "skills").glob("*/SKILL.md"))}
    if not sources:
        parser.error("No skills found. Run this script from the complete checkout.")
    if args.list:
        print("\n".join(sources))
        return 0
    unknown = set(args.skills) - sources.keys()
    if unknown:
        parser.error("Unknown skills: " + ", ".join(sorted(unknown)))
    selected = list(dict.fromkeys(args.skills)) or list(sources)
    dest = args.dest.expanduser().resolve()
    if dest == ROOT / "skills" or ROOT / "skills" in dest.parents:
        parser.error("Choose a destination outside the source skills directory.")
    for name in selected:
        source = sources[name]
        if source.is_symlink() or any((source / marker).exists() for marker in (POINTER, RECEIPT)):
            parser.error(f"Unexpected installation marker or link at source: {source}")
        inventory(source)
    plan = {name: classify(sources[name], dest / name) for name in selected}
    for name, status in plan.items():
        print(f"{status:8} {name}")
    if args.check:
        return int(any(s != "current" for s in plan.values()))
    if "conflict" in plan.values():
        parser.error("Unmanaged name collision. Choose --dest or reconcile the listed skill directories first.")
    if "changed" in plan.values() and not args.update:
        parser.error("Managed copy differs. Inspect changes, then use --update to save a backup and install.")
    dest.mkdir(parents=True, exist_ok=True)
    for name, status in plan.items():
        if status != "current":
            install_one(sources[name], dest / name, status, dest.parent / "myskills-backups")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        sys.exit(str(error))
