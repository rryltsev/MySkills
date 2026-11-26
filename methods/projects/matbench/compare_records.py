"""Compare E/F/stress on an identical structure, atom order, units and tensor convention."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(cpu, gpu):
    count = cpu["atoms"]
    if type(count) is not int or count <= 0 or type(gpu["atoms"]) is not int or count != gpu["atoms"]:
        raise ValueError("A matching positive atom count is required")
    if not cpu["weights"] or cpu["weights"] != gpu["weights"]:
        raise ValueError("Weight fingerprints differ or are missing")
    errors = {}
    for field, shapes in (("energy_eV", [()]), ("force_eV_A", [(count, 3)]), ("stress_GPa", [(6,), (3, 3)])):
        reference = np.asarray(cpu[field], dtype=float)
        actual = np.asarray(gpu[field], dtype=float)
        if reference.shape not in shapes or actual.shape != reference.shape:
            raise ValueError(f"Unexpected shapes for {field}")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(actual)):
            raise ValueError(f"Nonfinite {field}")
        difference = float(np.max(np.abs(actual - reference)))
        if field == "energy_eV":
            errors["energy_eV_atom"] = difference / count
        else:
            errors[field] = difference
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("gpu", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8-sig")) for path in (args.cpu, args.gpu)]
    print(json.dumps(compare(*records), indent=2))
