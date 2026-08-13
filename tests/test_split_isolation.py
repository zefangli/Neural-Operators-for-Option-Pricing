"""The no-leakage guarantee: the 80/10/10 split is by unique TRADE DATE, and it
is time-ordered (all train dates < all val dates < all test dates).

This is the single assumption the paper's generalisation claim rests on, so it
gets a direct assert against both HDF5 files rather than a comment. Reads only
`date` (S10) and `split_id` (uint8) -- a few hundred MB, seconds, no GPU.

Run either way:
    conda run -n dl_new python -m pytest tests/ -q
    conda run -n dl_new python tests/test_split_isolation.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from _common import PROJECT_ROOT  # noqa: E402

DATA = PROJECT_ROOT / "wrds_data_2020-2025"
H5 = {t: DATA / ("deeponet_tensors_%s.h5" % t) for t in ("call", "put")}


def _load(path):
    with h5py.File(path, "r") as f:
        return f["date"][:], f["split_id"][:]


def _each():
    for name, path in H5.items():
        if not path.exists():                       # keep the suite runnable without the data
            print("SKIP %s (%s missing)" % (name, path))
            continue
        yield name, _load(path)


def test_split_ids_are_exactly_0_1_2():
    for name, (_, sid) in _each():
        vals = set(np.unique(sid).tolist())
        assert vals == {0, 1, 2}, "%s: split ids %s" % (name, sorted(vals))


def test_no_date_appears_in_two_splits():
    for name, (date, sid) in _each():
        per_split = [set(np.unique(date[sid == s]).tolist()) for s in (0, 1, 2)]
        for a, b, lbl in ((0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")):
            shared = per_split[a] & per_split[b]
            assert not shared, "%s: %d dates leak across %s (e.g. %s)" % (
                name, len(shared), lbl, sorted(shared)[:3])


def test_split_is_time_ordered():
    for name, (date, sid) in _each():
        # np.unique returns sorted; bytes dtype has no max/min ufunc, so index.
        d = [np.unique(date[sid == s]) for s in (0, 1, 2)]
        assert d[0][-1] < d[1][0], "%s: train (ends %s) overlaps val (starts %s)" % (
            name, d[0][-1], d[1][0])
        assert d[1][-1] < d[2][0], "%s: val (ends %s) overlaps test (starts %s)" % (
            name, d[1][-1], d[2][0])


def test_split_is_by_date_not_by_row():
    """Every row of a given date carries the same split id."""
    for name, (date, sid) in _each():
        order = np.argsort(date, kind="stable")
        ds, ss = date[order], sid[order]
        starts = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
        lo = np.minimum.reduceat(ss, starts)
        hi = np.maximum.reduceat(ss, starts)
        assert np.array_equal(lo, hi), "%s: %d dates span >1 split id" % (
            name, int((lo != hi).sum()))


def test_split_proportions_are_roughly_80_10_10_by_date():
    for name, (date, sid) in _each():
        n = [len(np.unique(date[sid == s])) for s in (0, 1, 2)]
        tot = sum(n)
        frac = [x / tot for x in n]
        assert abs(frac[0] - 0.8) < 0.02, "%s: train date share %.4f" % (name, frac[0])
        assert abs(frac[1] - 0.1) < 0.02, "%s: val date share %.4f" % (name, frac[1])
        assert abs(frac[2] - 0.1) < 0.02, "%s: test date share %.4f" % (name, frac[2])


def test_call_and_put_share_the_same_date_partition():
    """Both files must draw the split boundary at the same dates, otherwise the
    call and put test-set metrics are not comparable."""
    parts = {}
    for name, (date, sid) in _each():
        parts[name] = [set(np.unique(date[sid == s]).tolist()) for s in (0, 1, 2)]
    if len(parts) < 2:
        return
    for s, lbl in enumerate(("train", "val", "test")):
        assert parts["call"][s] == parts["put"][s], (
            "call/put %s date sets differ (%d vs %d dates)" % (
                lbl, len(parts["call"][s]), len(parts["put"][s])))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS %s" % f.__name__)
    print("%d/%d passed" % (len(fns), len(fns)))
