"""Unit tests for analysis/duplicate_inputs_all_splits.py on synthetic arrays.

No HDF5, no manifest, no network. What is pinned here is the arithmetic the
artifact reports: which rows form a duplicate group under each key, that the
date is what makes cross-split collisions impossible, that grouping is on exact
float32 bits (so two values that differ in the last bit are NOT duplicates), and
that the percentage is rows-in-duplicate-groups over rows -- not groups over
groups, and not (rows - groups) over rows.

Run: pytest tests/test_duplicate_inputs_all_splits.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from duplicate_inputs_all_splits import (  # noqa: E402
    cross_split_stats, duplicate_stats, group_ids, key_columns)


def cols(dates, trunk):
    """(date bytes, trunk float32) -> the two key column lists."""
    d = np.array(dates, dtype="S10")
    t = np.array(trunk, dtype=np.float32)
    return key_columns("input_key", d, t), key_columns("query_key", d, t)


# ------------------------------------------------------------ key construction

def test_input_key_is_date_plus_four_trunk_columns():
    d = np.array([b"2020-01-01"] * 3)
    t = np.arange(12, dtype=np.float32).reshape(3, 4)
    ik, qk = key_columns("input_key", d, t), key_columns("query_key", d, t)
    assert len(ik) == 5 and len(qk) == 4
    assert ik[0] is d
    for i in range(4):
        assert np.array_equal(ik[i + 1], t[:, i])
        assert np.array_equal(qk[i], t[:, i])


def test_same_query_on_two_dates_is_a_duplicate_only_under_query_key():
    """The whole point of the two keys: the date is the market state."""
    ik, qk = cols([b"2020-01-01", b"2020-01-02"], [[0.1, 0.5, 0.02, 0.01]] * 2)
    assert duplicate_stats(ik)["n_rows_in_dup_groups"] == 0
    assert duplicate_stats(qk)["n_rows_in_dup_groups"] == 2


def test_grouping_is_on_exact_float32_bits_not_a_tolerance():
    a = np.float32(0.1)
    b = np.nextafter(a, np.float32(1.0), dtype=np.float32)
    assert a != b and abs(float(a) - float(b)) < 1e-8
    ik, _ = cols([b"2020-01-01"] * 2, [[a, 0.5, 0.0, 0.0], [b, 0.5, 0.0, 0.0]])
    assert duplicate_stats(ik)["n_dup_groups"] == 0


def test_a_difference_in_any_single_trunk_column_separates_the_rows():
    base = [0.1, 0.5, 0.02, 0.01]
    for j in range(4):
        other = list(base)
        other[j] += 1.0
        ik, _ = cols([b"2020-01-01"] * 2, [base, other])
        assert duplicate_stats(ik)["n_dup_groups"] == 0, "column %d ignored" % j


# ----------------------------------------------------------------- grouping

def test_group_ids_counts_and_sizes():
    ik, _ = cols([b"d1", b"d1", b"d1", b"d2"],
                 [[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    _order, _gid, sizes = group_ids(ik)
    assert sorted(sizes.tolist()) == [1, 1, 2]     # (d1,1)x2, (d1,2), (d2,1)


def test_empty_input_is_zero_not_a_crash():
    empty = [np.array([], dtype="S10")] + [np.array([], dtype=np.float32)] * 4
    s = duplicate_stats(empty)
    assert s == {"n_rows": 0, "n_groups": 0, "n_dup_groups": 0,
                 "n_rows_in_dup_groups": 0, "pct_rows_in_dup_groups": 0.0,
                 "max_group_size": 0, "group_size_histogram": {}}


# ------------------------------------------------------- percentage arithmetic

def test_percentage_is_rows_in_duplicate_groups_over_rows():
    """4 rows: one triple + one singleton -> 3/4 = 75%, NOT 1/2 groups and NOT
    (4-2)/4 = 50% 'excess rows'."""
    ik, _ = cols([b"d1"] * 4, [[1, 0, 0, 0]] * 3 + [[2, 0, 0, 0]])
    s = duplicate_stats(ik)
    assert (s["n_rows"], s["n_groups"], s["n_dup_groups"]) == (4, 2, 1)
    assert s["n_rows_in_dup_groups"] == 3
    assert s["pct_rows_in_dup_groups"] == pytest.approx(75.0)
    assert s["max_group_size"] == 3
    assert s["group_size_histogram"] == {"1": 1, "3": 1}


def test_all_distinct_gives_exactly_zero_percent():
    ik, _ = cols([b"d%d" % i for i in range(5)],
                 [[i, 0, 0, 0] for i in range(5)])
    s = duplicate_stats(ik)
    assert s["pct_rows_in_dup_groups"] == 0.0 and s["n_dup_groups"] == 0


def test_all_identical_gives_exactly_hundred_percent():
    ik, _ = cols([b"d1"] * 7, [[1, 2, 3, 4]] * 7)
    s = duplicate_stats(ik)
    assert s["pct_rows_in_dup_groups"] == pytest.approx(100.0)
    assert s["n_groups"] == 1 and s["max_group_size"] == 7


# ------------------------------------------------------------- cross-split

def test_input_key_cannot_span_splits_when_dates_do_not():
    """The date split makes this zero by construction; the test states it."""
    dates = [b"2020-01-01", b"2020-01-01", b"2021-01-01", b"2022-01-01"]
    trunk = [[1, 0, 0, 0]] * 4                      # identical query everywhere
    ik, qk = cols(dates, trunk)
    split = np.array([0, 0, 1, 2])
    ck = cross_split_stats(ik, split)
    assert ck["n_groups_spanning_splits"] == 0
    assert ck["n_rows_in_spanning_groups"] == 0
    assert ck["pct_rows_in_spanning_groups"] == 0.0
    # the same rows DO collide once the date is dropped -- diagnostic, not leakage
    cq = cross_split_stats(qk, split)
    assert cq["n_groups"] == 1 and cq["n_groups_spanning_splits"] == 1
    assert cq["n_rows_in_spanning_groups"] == 4


def test_cross_split_pairs_are_reported_separately():
    dates = [b"a", b"b", b"c"]
    _ik, qk = cols(dates, [[1, 0, 0, 0], [1, 0, 0, 0], [9, 0, 0, 0]])
    c = cross_split_stats(qk, np.array([0, 2, 1]))   # train + test share the key
    assert c["n_groups_by_split_pair"] == {"train_val": 0, "train_test": 1, "val_test": 0}
    assert c["n_groups_spanning_splits"] == 1


def test_a_key_present_in_all_three_splits_counts_in_every_pair():
    _ik, qk = cols([b"a", b"b", b"c"], [[1, 0, 0, 0]] * 3)
    c = cross_split_stats(qk, np.array([0, 1, 2]))
    assert c["n_groups_by_split_pair"] == {"train_val": 1, "train_test": 1, "val_test": 1}
    assert c["n_rows_in_spanning_groups"] == 3
    assert c["pct_rows_in_spanning_groups"] == pytest.approx(100.0)


def test_per_split_rates_are_computed_within_the_split_only():
    """A key repeated across splits must not inflate any split's own rate."""
    dates = [b"a", b"b", b"c", b"c"]
    _ik, qk = cols(dates, [[1, 0, 0, 0]] * 3 + [[1, 0, 0, 0]])
    split = np.array([0, 1, 2, 2])
    per = {nm: duplicate_stats([c[split == sid] for c in qk])
           for nm, sid in (("train", 0), ("val", 1), ("test", 2))}
    assert per["train"]["pct_rows_in_dup_groups"] == 0.0
    assert per["val"]["pct_rows_in_dup_groups"] == 0.0
    assert per["test"]["pct_rows_in_dup_groups"] == pytest.approx(100.0)
    # ... while the full-file rate sees all four rows as one group
    assert duplicate_stats(qk)["pct_rows_in_dup_groups"] == pytest.approx(100.0)
