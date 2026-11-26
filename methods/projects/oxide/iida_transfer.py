"""Withhold Iida et al. (2000) and their composition groups at every T and P.

Usage: python scripts/iida_transfer.py path/to/gpvisc_database.zip
"""
import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import composition_curve as curve
import audit_dataset as audit
import h5py
import numpy as np
from scipy.spatial.distance import cdist


def composition_key(row):
    return tuple(np.round(row / row.sum(), 9))


def prepare(curated_zip, hdf):
    lookup, _, slag_refs = audit.get_curated(curated_zip)
    with zipfile.ZipFile(curated_zip) as archive:
        held = []
        for kind in ("lp", "hp"):
            for row in audit.read_rows(archive, f"gpvisc_database_{kind}.csv"):
                if row["reference"] == audit.IIDA:
                    held.append((f"{kind}:{row['Index']}", row))
    exact = {composition_key(np.array([float(r[name]) for name in audit.OXIDES])) for _, r in held}
    mapping = defaultdict(list)
    linked_groups = set()
    all_keys = Counter()
    with h5py.File(hdf) as file:
        data = {s: audit.get_split(file, s) for s in audit.SPLITS}
    for split, values in data.items():
        mapped, keys, _ = audit.audit_split(values, split, lookup, slag_refs)
        all_keys.update(keys)
        for row in mapped:
            if row["iida_possible"]:
                linked_groups.add(row["composition_group"])
                for candidate in row["raw_record_candidates"].split("|"):
                    mapping[candidate].append((split, row["row"], row["composition_group"]))
    if all_keys != Counter({k: len(v) for k, v in lookup.items()}):
        raise ValueError("Raw and archived curated measurements differ")
    if any(record not in mapping for record, _ in held):
        raise ValueError("Some Iida raw records are missing from the published split")

    order = [audit.OXIDES.index(name) for name in curve.OXIDES]
    retained = {}
    counts = {}
    for split in ("train", "valid"):
        values = data[split]
        names = np.array([n.decode() for n in values["Name"]])
        keys = [composition_key(x) for x in values["x"]]
        keep = np.array([nm not in linked_groups and key not in exact for nm, key in zip(names, keys)])
        retained[split] = (values["x"][keep][:, order], values["P"][keep], values["T"][keep], values["y"][keep])
        counts[split] = {"original": len(names), "retained": int(keep.sum()), "excluded": int((~keep).sum()),
                         "retained_iida_group_overlap": len(set(names[keep]) & linked_groups),
                         "retained_iida_exact_composition_overlap": len({k for k, flag in zip(keys, keep) if flag} & exact)}
        if counts[split]["retained_iida_group_overlap"] or counts[split]["retained_iida_exact_composition_overlap"]:
            raise ValueError(f"Iida composition leakage into {split}")
    x = np.array([[float(r[name]) for name in curve.OXIDES] for _, r in held])
    x /= x.sum(axis=1, keepdims=True)
    p, t, y = (np.array([float(r[name]) for _, r in held]) for name in ("P", "T", "viscosity"))
    source_refs = Counter()
    info = []
    for record, _ in held:
        matches = mapping[record]
        # Retain every candidate location at identical measured input/target.
        src = "|".join(sorted({s for s, _, _ in matches}))
        source_refs[src] += 1
        info.append([record, "|".join(f"{s}:{i}" for s, i, _ in matches),
                     "|".join(sorted({n for _, _, n in matches}))])
    stats = {"source": audit.IIDA, "raw_holdout_n": len(held), "holdout_exact_compositions": len(exact),
             "holdout_published_group_tokens": len(linked_groups), "raw_records_by_original_split": dict(source_refs),
             "original_test_rows_used_in_training": 0, "fit_and_validation_counts": counts}
    return retained, (x, p, t, y), info, stats


def linear_control(train, held):
    def features(data):
        return np.column_stack([data[2], data[1], data[0]])
    x = features(train)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    mean = train[3].mean()
    beta = np.linalg.lstsq((x - center) / scale, train[3] - mean, rcond=None)[0]
    return mean + ((features(held) - center) / scale) @ beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curated_zip", type=Path)
    parser.add_argument("--hdf", type=Path, default=Path("upstream/charlesll-gpvisc-5dbe6cc/code_paper_EPSL/additional_data/all_viscosity.hdf5"))
    parser.add_argument("--out", type=Path, default=curve.RESULTS / "iida_transfer")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    retained, held, info, stats = prepare(args.curated_zip, args.hdf)
    print(json.dumps(stats, indent=2), flush=True)
    selected = curve.select_arrays(retained["train"], retained["valid"])
    curve.write_json(args.out / "validation.json", selected)
    merged = tuple(np.concatenate(arrays) for arrays in zip(retained["train"], retained["valid"]))
    model = curve.fit(*merged, **selected["selected"]["settings"])
    correction = selected["selected_correction"]["settings"]
    if correction is not None:
        curve.add_correction(model, *merged, **correction)
    pred = curve.predict(model, *held[:3])
    baseline = linear_control(merged, held)
    # L-infinity distance in mol% to the nearest retained composition captures
    # how close some withheld chemistries still are to retained measurements.
    distances = 100 * cdist(held[0], np.unique(merged[0], axis=0), metric="chebyshev").min(axis=1)
    stats.update(model=curve.measures(held[3], pred), linear_control=curve.measures(held[3], baseline),
                 nearest_composition_distance_mol_pct={"metric": "maximum absolute component difference; per measurement",
                                                       "min": float(distances.min()), "median": float(np.median(distances)),
                                                       "max": float(distances.max()), "under_0.1_mol_pct": int(sum(distances < 0.1))},
                 validation_rmse=selected["selected_correction"]["validation"]["rmse"])
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["raw_record", "original_split_rows", "composition_groups", "P_GPa", "T_K",
                         "log10_eta", "curve_prediction", "linear_prediction", "nearest_composition_mol_pct"])
        for i, row in enumerate(info):
            writer.writerow([*row, held[1][i], held[2][i], held[3][i], pred[i], baseline[i], distances[i]])
    curve.write_json(args.out / "metrics.json", stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
