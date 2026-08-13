"""Regenerate DATASET_MANIFEST.json -- the single source of truth for which
dataset is canonical, what filtering it carries, and each file's sha256.

Everything is measured from the HDF5 files themselves; nothing is hard-coded
except the prose status lines. Read-only apart from the manifest it writes.

    python make_dataset_manifest.py
    python make_dataset_manifest.py --check     # verify hashes, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL = "v5"
VIX_CSV = "vix_2015_2025.csv"

STATUS = {
    "v4": "unfiltered robustness sample; noncanonical for training",
    "v5": "canonical for training",
}
QUOTE_FILTER = {"v4": (None, None), "v5": ("static_bounds_midpoint", 0.0)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _s(attrs, key, default=""):
    v = attrs.get(key, default)
    return v.decode() if isinstance(v, bytes) else str(v)


def describe(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        dates = f["date"][:].astype("U10")
        split = f["split_id"][:]
        splits = {}
        for sid, nm in ((0, "train"), (1, "val"), (2, "test")):
            u = np.unique(dates[split == sid])
            splits[nm] = {"rows": int((split == sid).sum()), "n_dates": int(len(u)),
                          "first": str(u[0]) if len(u) else "",
                          "last": str(u[-1]) if len(u) else ""}
        rows = int(len(dates))
        attrs = dict(f.attrs)
    return {"path": path.name, "sha256": sha256(path), "rows": rows, "splits": splits}, attrs


def _shared(attrs_by_type: dict, key: str, default="") -> str:
    """Read a dataset-level attr that must agree across call and put.

    Anything type-specific (lineage hashes, row counts) must NOT come through
    here -- it gets its own per-type record.  Disagreement is a real defect in
    the export, so surface it instead of silently picking one side.
    """
    vals = {t: _s(a, key, default) for t, a in attrs_by_type.items()}
    if len(set(vals.values())) > 1:
        raise SystemExit(f"attr {key!r} differs across option types: {vals}")
    return next(iter(vals.values()))


def build(versions=("v4", "v5")) -> dict:
    man = {"canonical": CANONICAL,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "datasets": {}}
    for ver in versions:
        files, attrs = {}, {}
        for t in ("call", "put"):
            p = SCRIPT_DIR / f"deeponet_tensors_{t}_{ver}.h5"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            files[t], attrs[t] = describe(p)
        qf, tol = QUOTE_FILTER[ver]
        man["datasets"][ver] = {
            "dataset_version": ver,
            "schema_version": _shared(attrs, "schema_version"),
            "status": STATUS[ver],
            "quote_filter": qf,
            "quote_filter_tolerance": tol,
            "quote_filter_bounds": _shared(attrs, "quote_filter_bounds") or None,
            "maturity_rule": {
                "min_maturity_days": float(_shared(attrs, "export_min_maturity_days", "nan")),
                "t_basis": _shared(attrs, "export_t_basis"),
                "description": "strictly more than 24 hours to settlement",
            },
            "files": files,
            "source_hashes": {"vix_csv": _shared(attrs, "phase1_vix_sha256")},
        }
        if ver == "v5":
            man["datasets"][ver]["lineage"] = {
                "parquet_parts": _shared(attrs, "lineage_parquet_parts"),
                **{t: {"derived_from_v4_h5": _s(a, "lineage_v4_h5"),
                       "derived_from_v4_h5_sha256": _s(a, "lineage_v4_h5_sha256")}
                   for t, a in attrs.items()},
            }
    return man


def main():
    ap = argparse.ArgumentParser(description="Regenerate DATASET_MANIFEST.json")
    ap.add_argument("--check", action="store_true",
                    help="Recompute and diff against the existing manifest; write nothing.")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "DATASET_MANIFEST.json"))
    args = ap.parse_args()

    man = build()
    out = Path(args.out)
    if args.check:
        old = json.loads(out.read_text())
        drift = [f"{v}/{t}" for v in man["datasets"] for t in ("call", "put")
                 if old["datasets"][v]["files"][t]["sha256"] != man["datasets"][v]["files"][t]["sha256"]]
        print("DRIFT: " + ", ".join(drift) if drift else "all hashes match the manifest")
        raise SystemExit(1 if drift else 0)
    out.write_text(json.dumps(man, indent=2))
    for v, d in man["datasets"].items():
        for t, fi in d["files"].items():
            print(f"{v} {t}: {fi['rows']:>10,d} rows  {fi['sha256']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
