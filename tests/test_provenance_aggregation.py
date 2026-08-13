"""A cross-sample (v3/v4/v5) or multi-seed mixture must never reach a manuscript table.

`aggregate_results.collect()` used to sweep every non-smoke, non-quarantined
metrics.json with no filter on dataset version or seed, and `eval_to_json.py`
recorded neither. Once the 12 v4 runs exist, `--all` would also re-evaluate the 8
historical v3 runs and a seed-43 replication would be appended straight into T1 --
the exact v3/v4 confound the rebuild exists to eliminate.

These tests build synthetic metrics.json trees in a tempfile dir (nothing under
the real train_model_v3/ is created or read) and prove the guards hold.

Run: pytest tests/test_provenance_aggregation.py  or  python tests/test_provenance_aggregation.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import _common  # noqa: E402
import aggregate_results as agg  # noqa: E402

# Synthetic hashes: the point of the manifest rewrite is that NO hash is hard-coded
# in the code under test, so the tests are free to invent them.
SHA = {("v4", "call"): "a4" * 32, ("v4", "put"): "b4" * 32,
       ("v5", "call"): "a5" * 32, ("v5", "put"): "b5" * 32,
       ("v3", "call"): "a3" * 32, ("v3", "put"): "b3" * 32}
BRANCHES = ["branch_u", "spot_history", "vix_history"]
DIRNAME = {"branch_u": "vol_surface", "spot_history": "spot_history", "vix_history": "vix_history"}
CELLS = [(o, b, a) for o in ("call", "put") for b in BRANCHES for a in ("A", "B")]


def manifest_dict(canonical="v5", versions=("v3", "v4", "v5")):
    """The shape wrds_data_2020-2025/DATASET_MANIFEST.json carries."""
    status = {"v3": "superseded", "v4": "unfiltered robustness sample; noncanonical for training",
              "v5": "canonical for training"}
    return {
        "canonical": canonical,
        "generated": "2026-08-12T00:00:00Z",
        "datasets": {
            v: {
                "dataset_version": v,
                # NOTE: v5 keeps SCHEMA v4 -- the two axes are independent.
                "schema_version": "v3" if v == "v3" else "v4",
                "status": status[v],
                "quote_filter": "static_bounds_midpoint" if v == "v5" else None,
                "quote_filter_tolerance": 0.0 if v == "v5" else None,
                "maturity_rule": {"min_maturity_days": 1.0, "t_basis": "settlement"},
                "files": {ot: {"path": f"wrds_data_2020-2025/deeponet_tensors_{ot}_{v}.h5",
                               "sha256": SHA[(v, ot)], "rows": 10,
                               "splits": {"train": {"rows": 8, "n_dates": 8}}}
                          for ot in ("call", "put")},
                "source_hashes": {"vix_csv": "c" * 64},
            } for v in versions
        },
    }


def write_manifest(root, **kw):
    """Point _common at a synthetic manifest inside `root`; returns its path."""
    p = root / "DATASET_MANIFEST.json"
    p.write_text(json.dumps(manifest_dict(**kw)))
    _common.MANIFEST_PATH = p
    _common._MANIFEST_CACHE.clear()
    return p


@pytest.fixture(autouse=True)
def _isolated_manifest(tmp_path):
    """Every test runs against its own manifest; the real one is never read."""
    old = _common.MANIFEST_PATH
    write_manifest(tmp_path)
    yield tmp_path
    _common.MANIFEST_PATH = old
    _common._MANIFEST_CACHE.clear()


def metrics(option_type, branch, arch, seed=42, version="v5", canonical=True):
    return {
        "n_test": 1000, "r2_price": 0.9998, "r2_log_filtered": 0.99,
        "r2_log_full": 0.89, "rmse_log": 0.30, "rmse_price": 0.001, "mae_price": 0.0005,
        "option_type": option_type, "branch_key": branch, "arch": arch,
        "seed": seed, "dataset_version": version,
        # v4 and v5 share schema "v4": the table must key off dataset_version.
        "schema_version": "v3 (attr absent)" if version == "v3" else "v4",
        "h5_path": f"/data/deeponet_tensors_{option_type}_{version}.h5",
        "h5_sha256": SHA[(version, option_type)],
        "canonical_dataset": canonical,
    }


def write_run(root, m, dirname=None, subdir=None, drop=()):
    """Write one synthetic results_*/metrics.json into the fake train_model_v3/."""
    sub = subdir or m["option_type"]
    name = dirname or ("results_%s%s_%s%s" % (
        DIRNAME[m["branch_key"]], "_don" if m["arch"] == "B" else "",
        m["dataset_version"],
        "" if m["seed"] == 42 else "_seed%d" % m["seed"]))
    d = root / "train_model_v3" / sub / name
    d.mkdir(parents=True, exist_ok=True)
    m = {k: v for k, v in m.items() if k not in drop}
    (d / "metrics.json").write_text(json.dumps(m))
    return d


def build(runs):
    """Fake repo root with the given metrics rows; returns (root, results_dir)."""
    root = Path(tempfile.mkdtemp(prefix="prov_agg_"))
    for r in runs:
        write_run(root, r)
    (root / "results").mkdir()
    return root


def run_aggregate(root, version="v5", allow_partial=False):
    """Point aggregate_results at the fake tree and run main()."""
    old_root, old_out, old_argv = agg.PROJECT_ROOT, agg.OUT_DIR, sys.argv
    agg.PROJECT_ROOT, agg.OUT_DIR = root, root / "results"
    sys.argv = ["aggregate_results.py", "--dataset-version", version]
    if allow_partial:
        sys.argv.append("--allow-partial")
    try:
        agg.main()
    finally:
        agg.PROJECT_ROOT, agg.OUT_DIR, sys.argv = old_root, old_out, old_argv


def full_v5(version="v5"):
    return [metrics(*c, version=version) for c in CELLS]


full_v4 = full_v5   # older name; the canonical sample is now v5


# ---------------------------------------------------------------- accept

def test_clean_12_cell_v5_set_is_accepted():
    root = build(full_v5())
    try:
        run_aggregate(root)
        csv_text = (root / "results" / "all_metrics_v5.csv").read_text()
        assert csv_text.count("branch_u") == 4          # 2 option types x 2 archs
        assert len((root / "results" / "table_T1_v5.md").read_text().strip().splitlines()) == 14
        for stale in ("all_metrics.csv", "all_metrics_v4.csv", "table_T1_v4.md"):
            assert not (root / "results" / stale).exists(), f"must not clobber {stale}"
    finally:
        shutil.rmtree(root)


def test_v4_set_is_still_aggregable_as_the_robustness_sample():
    """v4 is noncanonical but not forbidden -- it writes its own _v4 files."""
    root = build(full_v5(version="v4"))
    try:
        run_aggregate(root, version="v4")
        assert (root / "results" / "all_metrics_v4.csv").exists()
        assert not (root / "results" / "all_metrics_v5.csv").exists()
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------- reject

def test_v4_rows_are_rejected_from_the_canonical_v5_table_by_dataset_version():
    """Canonical is v5: a v4-sample run must not silently fill a v5 cell.

    The two share schema_version "v4", so the refusal has to name the DATASET
    version -- naming the schema would be meaningless here.
    """
    runs = full_v5()
    runs[0] = metrics("call", "branch_u", "A", version="v4")
    root = build(runs)
    try:
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        msg = str(e.value)
        assert "MISSING" in msg and "call/vol_surface/A" in msg
        assert "dataset_version=v4" in msg and "want v5" in msg
    finally:
        shutil.rmtree(root)


def test_v4_v5_mixture_is_rejected():
    """Half the cells from each sample: neither table can be completed."""
    runs = full_v5()[:6] + [metrics(*c, version="v4") for c in CELLS[6:]]
    root = build(runs)
    try:
        for ver in ("v5", "v4"):
            with pytest.raises(SystemExit) as e:
                run_aggregate(root, version=ver)
            assert "MISSING" in str(e.value)
        assert not list((root / "results").glob("table_T1_*"))
    finally:
        shutil.rmtree(root)


def test_v3_v5_mixture_is_rejected_and_names_the_cell():
    """One cell evaluated on the v3 sample: it is dropped, so T1 is incomplete."""
    runs = full_v5()
    runs[0] = metrics("call", "branch_u", "A", version="v3")
    root = build(runs)
    try:
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        assert "MISSING" in str(e.value) and "call/vol_surface/A" in str(e.value)
    finally:
        shutil.rmtree(root)


def test_missing_cell_is_rejected_and_named():
    runs = [m for m in full_v4() if (m["option_type"], m["branch_key"], m["arch"])
            != ("put", "vix_history", "B")]
    root = build(runs)
    try:
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        msg = str(e.value)
        assert "MISSING" in msg and "put/vix_history/B" in msg
        assert "call/vol_surface/A" not in msg, "only the missing cell may be named"
    finally:
        shutil.rmtree(root)


def test_duplicate_cell_is_rejected_and_named():
    root = build(full_v4())
    try:
        write_run(root, metrics("call", "branch_u", "A"), dirname="results_vol_surface_v5_rerun")
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        msg = str(e.value)
        assert "DUPLICATE" in msg and "call/vol_surface/A" in msg
        assert "results_vol_surface_v5_rerun" in msg, "the offending file must be named"
    finally:
        shutil.rmtree(root)


def test_mixed_hdf5_hash_is_rejected():
    runs = full_v4()
    runs[0] = dict(runs[0], h5_sha256="d" * 64)
    root = build(runs)
    try:
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        assert "MIXED call HDF5 sha256" in str(e.value)
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("drop", [("dataset_version",), ("seed",)])
def test_rows_without_provenance_never_silently_count(drop):
    runs = full_v4()[1:]
    root = build(runs)
    try:
        write_run(root, metrics("call", "branch_u", "A"), drop=drop)
        with pytest.raises(SystemExit) as e:
            run_aggregate(root)
        assert "call/vol_surface/A" in str(e.value)
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------- seeds

def test_seed43_stays_out_of_T1_but_enters_replication():
    runs = full_v4() + [metrics("call", "branch_u", "A", seed=43),
                        metrics("call", "branch_u", "A", seed=44)]
    root = build(runs)
    try:
        run_aggregate(root)
        t1 = (root / "results" / "table_T1_v5.md").read_text()
        assert len(t1.strip().splitlines()) == 14, "T1 must stay at exactly 12 rows"
        assert "43" not in (root / "results" / "all_metrics_v5.csv").read_text().replace(
            SHA[("v5", "call")], "").replace(SHA[("v5", "put")], "")
        repl = (root / "results" / "replication_seeds_v5.md").read_text()
        assert "42,43,44" in repl
        rows = (root / "results" / "replication_seeds_v5.csv").read_text()
        assert "3" in rows  # n_seeds
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------- exclusions

def test_vvix_and_smoke_rows_stay_excluded():
    root = build(full_v4())
    try:
        # quarantined VVIX artifact (location, not name) + a 1-batch smoke run
        write_run(root, metrics("call", "vix_history", "A"), subdir="_vix_is_vvix_LEGACY",
                  dirname="results_vix_history")
        write_run(root, metrics("put", "branch_u", "B"),
                  dirname="results_vol_surface_don_v5_smoke")
        # right schema+sample label but a non-manifest (smoke-built) HDF5, inside a
        # normal-looking dir
        write_run(root, metrics("put", "spot_history", "A", canonical=False),
                  dirname="results_spot_history_v5_smokedata")
        run_aggregate(root)   # still exactly the 12 canonical cells -> no SystemExit
        csv_text = (root / "results" / "all_metrics_v5.csv").read_text()
        for bad in ("_vix_is_vvix_LEGACY", "_v5_smoke", "smokedata"):
            assert bad not in csv_text, f"{bad} row reached the table"
    finally:
        shutil.rmtree(root)


# ------------------------------------------- eval_to_json provenance rejection

def test_eval_rejects_config_hdf5_schema_mismatch():
    """config.json says v4 but the resolved HDF5 has no schema_version -> loud fail."""
    h5py = pytest.importorskip("h5py")
    import eval_to_json as ev
    root = Path(tempfile.mkdtemp(prefix="prov_eval_"))
    try:
        p = root / "fake_v3.h5"
        with h5py.File(p, "w") as f:
            f["x"] = [1.0]
        cfg = {"option_type": "call", "seed": 42, "h5_sha256": "a" * 64,
               "data_provenance": {"schema_version": "v4"}}
        with pytest.raises(SystemExit) as e:
            ev.run_provenance(cfg, p)
        assert "provenance mismatch" in str(e.value)

        with h5py.File(p, "a") as f:
            f.attrs["schema_version"] = "v4"
        prov = ev.run_provenance(cfg, p)          # reuses the cached hash, no rehash
        assert prov["dataset_version"] == "v4" and prov["seed"] == 42
        assert prov["canonical_dataset"] is False  # hash is in no manifest entry
        assert prov["is_canonical_version"] is False  # canonical is v5
    finally:
        shutil.rmtree(root)


def test_eval_identifies_v5_by_manifest_hash_not_by_schema():
    """A v5 file carries schema_version 'v4'. Identity must come from the manifest."""
    h5py = pytest.importorskip("h5py")
    import eval_to_json as ev
    root = Path(tempfile.mkdtemp(prefix="prov_eval_v5_"))
    try:
        p = root / "fake_v5.h5"
        with h5py.File(p, "w") as f:
            f["x"] = [1.0]
            f.attrs["schema_version"] = "v4"        # SAME schema as v4
            f.attrs["dataset_version"] = "v5"       # different SAMPLE
        cfg = {"option_type": "call", "seed": 42, "h5_sha256": SHA[("v5", "call")],
               "data_provenance": {"schema_version": "v4", "dataset_version": "v5"}}
        prov = ev.run_provenance(cfg, p)
        assert prov["dataset_version"] == "v5"
        assert prov["schema_version"] == "v4", "schema is reported as-is, not rewritten"
        assert prov["canonical_dataset"] is True and prov["is_canonical_version"] is True

        # same schema, but the config claims the other sample -> refuse
        bad = dict(cfg, data_provenance={"schema_version": "v4", "dataset_version": "v4"})
        with pytest.raises(SystemExit) as e:
            ev.run_provenance(bad, p)
        assert "dataset (SAMPLE) mismatch" in str(e.value)
    finally:
        shutil.rmtree(root)


def test_v4_file_is_identified_as_v4_by_hash_even_without_a_dataset_version_attr():
    """Every v4 build predates the attr; the manifest hash still identifies it."""
    h5py = pytest.importorskip("h5py")
    import eval_to_json as ev
    root = Path(tempfile.mkdtemp(prefix="prov_eval_v4_"))
    try:
        p = root / "fake_v4.h5"
        with h5py.File(p, "w") as f:
            f["x"] = [1.0]
            f.attrs["schema_version"] = "v4"
        cfg = {"option_type": "put", "seed": 42, "h5_sha256": SHA[("v4", "put")],
               "data_provenance": {"schema_version": "v4"}}
        prov = ev.run_provenance(cfg, p)
        assert prov["dataset_version"] == "v4"
        assert prov["canonical_dataset"] is True      # canonical v4 FILE ...
        assert prov["is_canonical_version"] is False  # ... but v5 is the canonical SAMPLE
    finally:
        shutil.rmtree(root)


# ------------------------------------------------------- manifest is required

def test_missing_manifest_fails_clearly(tmp_path):
    _common.MANIFEST_PATH = tmp_path / "nope" / "DATASET_MANIFEST.json"
    _common._MANIFEST_CACHE.clear()
    with pytest.raises(SystemExit) as e:
        _common.canonical_version()
    assert "MISSING DATASET MANIFEST" in str(e.value)
    with pytest.raises(SystemExit):
        _common.dataset_h5("v5", "call")


def test_unknown_dataset_version_fails_clearly(tmp_path):
    write_manifest(tmp_path, versions=("v3", "v4"))    # no v5 in this manifest
    with pytest.raises(SystemExit) as e:
        _common.dataset_h5("v5", "call")
    msg = str(e.value)
    assert "UNKNOWN dataset-version 'v5'" in msg and "v3, v4" in msg


def test_canonical_version_comes_from_the_manifest(tmp_path):
    write_manifest(tmp_path, canonical="v4")
    assert _common.canonical_version() == "v4"
    write_manifest(tmp_path, canonical="v5")
    assert _common.canonical_version() == "v5"


def test_dataset_and_schema_version_are_independent(tmp_path):
    """v5 is a different SAMPLE with the SAME tensor schema."""
    assert _common.dataset_versions("v5") == ("v5", "v4")
    assert _common.dataset_versions("v4") == ("v4", "v4")
    assert _common.dataset_versions("v3") == ("v3", "v3")


def test_no_hard_coded_hash_or_canonical_version_remains():
    src = (Path(__file__).resolve().parent.parent / "analysis" / "eval_to_json.py").read_text()
    assert "CANONICAL_V4_SHA256" not in src
    import re
    assert not re.search(r"[0-9a-f]{64}", src), "a dataset hash is hard-coded again"


# ------------------------------------------------- quote-liquidity REPORT strata

def test_liquidity_strata_are_reported_and_drop_nothing():
    np = pytest.importorskip("numpy")
    from _common import compute_metrics

    n = 200
    rng = np.random.default_rng(0)
    target = rng.normal(-2.0, 1.0, n)
    pred = target + rng.normal(0, 0.01, n)
    T = np.full(n, 0.5)
    bid = np.where(np.arange(n) < 50, 0.0, 1.0)          # 50 zero-bid rows
    ask = np.where(np.arange(n) % 5 == 0, 100.0, 1.2)    # 40 very wide quotes

    base = compute_metrics(pred, target, T)
    m = compute_metrics(pred, target, T, best_bid=bid, best_offer=ask)

    # 1. the headline numbers are computed on ALL rows and are unchanged
    assert m["n_test"] == n == base["n_test"]
    for k in ("r2_price", "r2_log_full", "r2_log_filtered", "rmse_log"):
        assert m[k] == base[k], "liquidity strata must not alter the full-sample metrics"

    s = m["liquidity_strata"]
    assert set(s) == {"bid_pos", "relspread_le1", "bid_pos_and_relspread_le1"}
    assert s["bid_pos"]["n_test"] == 150
    assert s["bid_pos_and_relspread_le1"]["n_test"] <= s["bid_pos"]["n_test"] < n
    assert "no row is dropped" in m["liquidity_strata_note"]["_not_a_filter"].lower()

    # 2. v3 (no bid/ask columns) simply reports no strata, it does not fail
    assert "liquidity_strata" not in compute_metrics(pred, target, T)


def test_relative_spread_threshold_is_bounded_by_two():
    """(ask-bid)/mid <= 2 whenever bid >= 0, so <=1 is a real cut, >2 is vacuous."""
    np = pytest.importorskip("numpy")
    from _common import liquidity_masks
    bid = np.array([0.0, 0.01, 1.0, 5.0])
    ask = np.array([100.0, 100.0, 1.2, 5.0])
    rel = (ask - bid) / (0.5 * (bid + ask))
    assert (rel <= 2.0 + 1e-12).all()
    masks = liquidity_masks(bid, ask)
    assert masks["bid_pos"].tolist() == [False, True, True, True]
    assert masks["relspread_le1"].tolist() == [False, False, True, True]
    assert liquidity_masks(None, ask) == {}, "v3 degrades to no strata, not an error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
