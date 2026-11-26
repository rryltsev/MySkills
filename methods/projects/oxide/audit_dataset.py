"""Audit the published GPVisc split against the user-supplied curated database ZIP.

The per-row table retains all possible source IDs when measurements repeat.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


OXIDES = "sio2 tio2 al2o3 feo fe2o3 mno na2o k2o mgo cao p2o5 h2o".split()
SPLITS = ("train", "valid", "test")
IIDA = "Iida et al. 2000"


def read_rows(archive, filename):
    with archive.open(filename) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))


def row_key(x, p, t, y):
    # Preserve more precision than the measurement tables while allowing for
    # floating-point normalization and differences between CSV readers.
    return (*np.round(x, 9), round(float(p), 7), round(float(t), 5), round(float(y), 7))


def get_curated(archive_path):
    lookup = defaultdict(list)
    summary = {}
    with zipfile.ZipFile(archive_path) as z:
        references = read_rows(z, "gpvisc_database_references.csv")
        slag_refs = {r["Reference"] for r in references if re.search(r"\bslags?\b", r["Details"], re.I)}
        for dataset in ("lp", "hp"):
            rows = read_rows(z, f"gpvisc_database_{dataset}.csv")
            features, refs = [], set()
            for row in rows:
                x = np.array([float(row[c]) for c in OXIDES])
                k = row_key(x / x.sum(), row["P"], row["T"], row["viscosity"])
                lookup[k].append((f"{dataset}:{row['Index']}", row["reference"]))
                features.append(k)
                refs.add(row["reference"])
            summary[dataset] = {
                "rows": len(rows), "distinct_reference_labels": len(refs),
                "repeat_rows_beyond_first_same_input": len(features) - len({k[:-1] for k in features}),
                "repeat_rows_beyond_first_same_input_and_target": len(features) - len(set(features)),
                "iida_rows": sum(row["reference"] == IIDA for row in rows),
                "slag_bibliography_keyword_rows": sum(row["reference"] in slag_refs for row in rows),
            }
    return lookup, summary, slag_refs


def get_split(hdf, subset):
    columns = [c.decode() for c in hdf["X_columns"][:].ravel()]
    data = {c: hdf[f"{c}_{subset}"][:].ravel() for c in ("P", "T", "y", "Name", "Sciglass")}
    data["x"] = hdf[f"X_{subset}"][:, [columns.index(o) for o in OXIDES]]
    return data


def audit_split(data, subset, lookup, slag_refs):
    mapped, matched_keys = [], []
    for index in np.flatnonzero(~data["Sciglass"]):
        key = row_key(data["x"][index], data["P"][index], data["T"][index], data["y"][index])
        matched_keys.append(key)
        candidates = lookup[key]
        if not candidates:
            raise ValueError(f"Unmapped curated row {subset}/{index}")
        references = sorted({ref for _, ref in candidates})
        mapped.append({
            "split": subset, "row": int(index),
            "composition_group": data["Name"][index].decode(),
            "raw_record_candidates": "|".join(record for record, _ in candidates),
            "source_candidates": "|".join(references),
            "iida_definite": references == [IIDA], "iida_possible": IIDA in references,
            "slag_title_definite": all(ref in slag_refs for ref in references),
            "slag_title_possible": any(ref in slag_refs for ref in references),
        })
    hp_mask = data["P"] > 0  # the LP CSV encodes ambient pressure as exactly zero
    summary = {
        "n": len(hp_mask), "n_atmospheric": int(sum(~hp_mask)),
        "n_high_pressure": int(sum(hp_mask)), "n_sciglass": int(sum(data["Sciglass"])),
        "n_curated": len(mapped),
        "unique_tokens_atmospheric": len(set(data["Name"][~hp_mask])),
        "unique_tokens_high_pressure": len(set(data["Name"][hp_mask])),
        "multiple_possible_raw_rows": sum("|" in r["raw_record_candidates"] for r in mapped),
        "multiple_possible_sources": sum("|" in r["source_candidates"] for r in mapped),
        **{c: sum(r[c] for r in mapped) for c in (
            "iida_definite", "iida_possible", "slag_title_definite", "slag_title_possible"
        )},
    }
    return mapped, matched_keys, summary


def shared_groups(train, other, mask_fn):
    train_mask, other_mask = mask_fn(train["P"]), mask_fn(other["P"])
    seen = set(train["Name"][train_mask])
    names = other["Name"][other_mask]
    return {
        "overlapping_tokens": len(seen.intersection(names)),
        "rows_with_training_token": sum(n in seen for n in names),
        "total_rows": len(names),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curated_zip", type=Path)
    parser.add_argument("--hdf", type=Path, default=Path("upstream/charlesll-gpvisc-5dbe6cc/src/gpvisc/data/all_viscosity.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    lookup, raw_stats, slag_refs = get_curated(args.curated_zip)
    expected_counts = Counter({k: len(v) for k, v in lookup.items()})
    observed_counts = Counter()
    split_stats, overlaps = {}, {}
    with h5py.File(args.hdf, "r") as hdf:
        data = {s: get_split(hdf, s) for s in SPLITS}
    args.output.mkdir(parents=True, exist_ok=True)
    for subset in SPLITS:
        mapped, keys, split_stats[subset] = audit_split(data[subset], subset, lookup, slag_refs)
        observed_counts.update(keys)
        if subset == "test":
            with (args.output / "test_curated_sources.csv").open("w", newline="", encoding="utf-8") as f:
                table = csv.DictWriter(f, fieldnames=list(mapped[0]))
                table.writeheader()
                table.writerows(mapped)
        if subset != "train":
            overlaps[subset] = {name: shared_groups(data["train"], data[subset], mask) for name, mask in (
                ("atmospheric_vs_atmospheric", lambda p: p == 0),
                ("high_pressure_vs_high_pressure", lambda p: p > 0),
                ("all_vs_all", lambda p: np.ones(len(p), dtype=bool)),
            )}
    if observed_counts != expected_counts:
        raise ValueError("The published split and local curated archive have different measurement multiplicities")
    report = {
        "curated_zip_sha256": hashlib.sha256(args.curated_zip.read_bytes()).hexdigest(),
        "hdf5_sha256": hashlib.sha256(args.hdf.read_bytes()).hexdigest(),
        "matched_all_curated_observations_with_multiplicity": True,
        "source_map": "All curated test rows. Index is zero-based within the archived HDF5 test split. Possible source and raw record IDs are pipe-separated. Slag bibliography flag uses a case-insensitive whole word 'slag' or 'slags' in the local references CSV Details field.",
        "raw": raw_stats, "split": split_stats, "composition_group_overlap_with_train": overlaps,
    }
    (args.output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
