"""v5 lineage must record BOTH source files, one per option type.

Regression test for the 2026-08-13 bug: make_dataset_manifest.build() reused a
single `attrs` variable across the call/put loop, so every per-type attribute
was overwritten by whichever type ran last (put).  The v5 lineage therefore
claimed a single `derived_from_v4_h5_sha256` -- put v4's hash -- and call's
provenance was silently dropped.

Run: pytest tests/test_manifest_lineage.py  or  python tests/test_manifest_lineage.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from _common import manifest_entry  # noqa: E402

# Hashes of the v4 HDF5 files v5 was derived from, verified 2026-08-13.
V4_SHA256 = {
    "call": "5f1c42aa2cc125fa0344b9e3d19fb374278aa629560c420c7ea4c76a6f910546",
    "put": "2ba5b13b0beb374d86790b6fc94799ea92dbd057454fca4af86f357a1bafbd29",
}


def _lineage():
    lin = manifest_entry("v5").get("lineage")
    assert lin, "v5 manifest entry has no lineage block"
    return lin


def test_lineage_is_not_flat():
    """The clobbered shape put one dataset-level hash where two belong."""
    assert "derived_from_v4_h5_sha256" not in _lineage(), (
        "v5 lineage still carries a single dataset-level source hash; it must "
        "record one per option type (call/put)"
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_lineage_records_the_right_v4_source(option_type):
    rec = _lineage().get(option_type)
    assert rec, f"v5 lineage has no {option_type} record"
    assert rec.get("derived_from_v4_h5_sha256") == V4_SHA256[option_type], (
        f"v5 {option_type} lineage hash is {rec.get('derived_from_v4_h5_sha256')!r}, "
        f"expected {V4_SHA256[option_type]}"
    )
    assert rec.get("derived_from_v4_h5") == f"deeponet_tensors_{option_type}_v4.h5"


def test_call_and_put_lineage_are_distinct():
    lin = _lineage()
    assert lin["call"]["derived_from_v4_h5_sha256"] != lin["put"]["derived_from_v4_h5_sha256"], \
        "call and put lineage hashes are identical -- one type's attrs clobbered the other"


def test_lineage_matches_the_v4_manifest_entry():
    """The recorded sources are the same files v4 itself is hashed as."""
    v4_files = manifest_entry("v4")["files"]
    for ot in ("call", "put"):
        assert _lineage()[ot]["derived_from_v4_h5_sha256"] == v4_files[ot]["sha256"], \
            f"v5 {ot} lineage does not match the v4 {ot} file hash in the same manifest"


if __name__ == "__main__":
    ok = True
    checks = [(test_lineage_is_not_flat, ()), (test_call_and_put_lineage_are_distinct, ()),
              (test_lineage_matches_the_v4_manifest_entry, ())]
    checks += [(test_lineage_records_the_right_v4_source, (ot,)) for ot in ("call", "put")]
    for fn, args in checks:
        try:
            fn(*args)
            print(f"PASS {fn.__name__} {args}")
        except Exception as exc:  # noqa: BLE001 - direct-run reporting
            print(f"FAIL {fn.__name__} {args}: {exc}")
            ok = False
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
