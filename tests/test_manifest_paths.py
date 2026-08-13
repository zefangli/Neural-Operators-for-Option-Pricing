"""Manifest file paths must resolve to files that actually exist.

Regression test for the 2026-08-12 integration bug: DATASET_MANIFEST.json lives
in wrds_data_2020-2025/ and records its files by bare filename, but the resolver
joined them against PROJECT_ROOT -- so every lookup pointed at
<repo>/deeponet_tensors_*.h5 and all three v5 CPU analyses died with
FileNotFoundError. Manifest paths resolve relative to the MANIFEST'S OWN
directory.

Run: pytest tests/test_manifest_paths.py  or  python tests/test_manifest_paths.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from _common import (MANIFEST_PATH, canonical_version, dataset_h5,  # noqa: E402
                     load_manifest, manifest_entry)

VERSIONS = ["v4", "v5"]
OPTION_TYPES = ["call", "put"]


def test_manifest_exists():
    assert MANIFEST_PATH.is_file(), f"manifest missing at {MANIFEST_PATH}"


def test_canonical_version_is_declared():
    man = load_manifest()
    assert man.get("canonical") in man.get("datasets", {}), \
        "manifest 'canonical' must name one of its own datasets"


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_dataset_path_resolves_to_existing_file(version, option_type):
    p = dataset_h5(version, option_type)
    assert p.is_absolute(), f"{version}/{option_type} resolved to a relative path: {p}"
    assert p.exists(), (
        f"{version}/{option_type} resolved to {p}, which does not exist. "
        "Manifest paths resolve relative to the manifest's own directory."
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_resolved_path_sits_beside_the_manifest(version, option_type):
    """Guards the specific regression: resolution against the repo root."""
    p = dataset_h5(version, option_type)
    assert p.parent == MANIFEST_PATH.parent, (
        f"{version}/{option_type} resolved outside the manifest's directory: {p.parent}"
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_manifest_records_a_sha_per_file(version):
    files = manifest_entry(version).get("files", {})
    for ot in OPTION_TYPES:
        sha = files.get(ot, {}).get("sha256", "")
        assert len(sha) == 64, f"{version}/{ot} sha256 is not 64 hex chars: {sha!r}"


def test_v5_is_canonical_and_filtered():
    """v5 is the quote-filtered sample; v4 is retained but noncanonical."""
    assert canonical_version() == "v5"
    v5, v4 = manifest_entry("v5"), manifest_entry("v4")
    assert v5.get("quote_filter") == "static_bounds_midpoint"
    assert float(v5.get("quote_filter_tolerance")) == 0.0
    assert v4.get("quote_filter") in (None, ""), "v4 must remain the UNFILTERED sample"


if __name__ == "__main__":
    ok = True
    for fn, args in [(test_manifest_exists, ()), (test_canonical_version_is_declared, ()),
                     (test_v5_is_canonical_and_filtered, ())]:
        try:
            fn(*args)
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - direct-run reporting
            print(f"FAIL {fn.__name__}: {exc}")
            ok = False
    for v in VERSIONS:
        for ot in OPTION_TYPES:
            for fn in (test_dataset_path_resolves_to_existing_file,
                       test_resolved_path_sits_beside_the_manifest):
                try:
                    fn(v, ot)
                    print(f"PASS {fn.__name__:44s} {v}/{ot}")
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {fn.__name__:44s} {v}/{ot}: {exc}")
                    ok = False
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
