"""Capture ASE predictions for an export diagnostic, or compare two saved captures."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hashed.update(block)
    return hashed.hexdigest()


def predict(args):
    from ase.io import iread
    from deepmd.calculator import DP

    suffix = ".pt" if args.role == "reference" else ".pt2"
    if args.model.suffix != suffix:
        raise ValueError(f"{args.role} requires a {suffix} file")
    model_hash = digest(args.model)
    input_hash = digest(args.frames)
    calculator = DP(model=str(args.model.resolve()))
    frames = []
    for atoms in iread(args.frames, index=":"):
        if not all(atoms.pbc) or atoms.get_volume() <= 0:
            raise ValueError("Use 3-D periodic cells of positive volume for this stress diagnostic.")
        atoms.calc = calculator
        frames.append(dict(n=len(atoms), energy=float(atoms.get_potential_energy(apply_constraint=False)),
                           forces=atoms.get_forces(apply_constraint=False).tolist(),
                           stress=atoms.get_stress(voigt=True, apply_constraint=False).tolist()))
    if not frames:
        raise ValueError("No input frames")
    if model_hash != digest(args.model) or input_hash != digest(args.frames):
        raise ValueError("Model or frame file changed during capture")
    data = dict(role=args.role, model_sha256=model_hash, frames_sha256=input_hash,
                packages={p: importlib.metadata.version(p) for p in ("ase", "deepmd-kit", "torch")},
                frames=frames)
    serialized = json.dumps(data, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as out:
        out.write(serialized + "\n")
    return 0


def components(frame):
    n = frame["n"]
    if type(n) is not int or n < 1:
        raise ValueError("Invalid atom count")
    forces, stress = frame["forces"], frame["stress"]
    if len(forces) != n or any(len(row) != 3 for row in forces) or len(stress) != 6:
        raise ValueError("Expected N x 3 forces and six ASE Voigt stress components")
    groups = [[frame["energy"]], [v for row in forces for v in row], stress]
    if any(type(v) not in (int, float) or not math.isfinite(v) for group in groups for v in group):
        raise ValueError("Every component must be finite")
    return groups


def compare(reference, exported, limits):
    if reference["role"] != "reference" or exported["role"] != "export":
        raise ValueError("Expected reference and export captures in that order")
    if reference["frames_sha256"] != exported["frames_sha256"]:
        raise ValueError("Input files differ. Use the same frame file and order.")
    rows, candidates = reference["frames"], exported["frames"]
    if not rows or len(rows) != len(candidates):
        raise ValueError("Frame counts differ or are zero")
    if any(not math.isfinite(x) or x < 0 for x in limits):
        raise ValueError("Tolerances must be finite and nonnegative")
    report = []
    for index, (a, b) in enumerate(zip(rows, candidates)):
        if a["n"] != b["n"]:
            raise ValueError(f"Different atom count in frame {index}")
        errors = [max(abs(x-y) for x, y in zip(left, right))
                  for left, right in zip(components(a), components(b))]
        errors[0] /= a["n"]
        report.append(dict(frame=index, dE_eV_atom=errors[0], max_dF_eV_A=errors[1],
                           max_dStress_eV_A3=errors[2], passed=all(x <= y for x, y in zip(errors, limits))))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cap = subs.add_parser("predict", help="capture raw static E/F/stress on the given model")
    cap.add_argument("role", choices=("reference", "export"))
    cap.add_argument("model", type=Path)
    cap.add_argument("frames", type=Path, help="ordered periodic configurations readable by ASE")
    cap.add_argument("output", type=Path, help="new JSON file")
    check = subs.add_parser("compare", help="compare saved captures on exactly the same input file")
    check.add_argument("reference", type=Path)
    check.add_argument("export", type=Path)
    for flag in ("energy", "forces", "stress"):
        check.add_argument("--" + flag, type=float, required=True)
    subs.add_parser("demo", help="run a synthetic comparison without ML libraries")
    args = parser.parse_args()
    if args.command == "predict":
        return predict(args)
    if args.command == "demo":
        sample = dict(n=2, energy=-1., forces=[[0., 0., 0.]] * 2, stress=[0.] * 6)
        left = dict(role="reference", frames_sha256="synthetic", frames=[sample])
        right = dict(left, role="export")
        limits = (1e-5, 2e-4, 1e-6)
        good = compare(left, right, limits)
        right["frames"] = [dict(sample, stress=[2e-6] + [0.] * 5)]
        bad = compare(left, right, limits)
        print(json.dumps(dict(equal=good, perturbed=bad), indent=2))
        return int(not good[0]["passed"] or bad[0]["passed"])
    a, b = [json.loads(path.read_text(encoding="utf-8")) for path in (args.reference, args.export)]
    result = compare(a, b, (args.energy, args.forces, args.stress))
    print(json.dumps(dict(reference_sha256=a["model_sha256"], export_sha256=b["model_sha256"],
                          frames_sha256=b["frames_sha256"], frames=result), indent=2, allow_nan=False))
    return int(not all(row["passed"] for row in result))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(str(error))
