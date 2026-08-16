"""Guards on the scalar-sigma ablation table (analysis/eval_ablation.py).

Two regressions this pins:
  (a) `--max-contracts` is a dev-sized subsample. It used to write the SAME
      filenames as the full run, so one smoke invocation silently replaced the
      manuscript's `results/ablation_scalar_sigma_v5.*` with a partial-sample
      table. Subsamples now go to `*_smoke` stems.
  (b) The four compared runs were only checked for a single dataset_version.
      Off-seed runs, mixed HDF5 hashes, mixed quote filters, a non-canonical
      (smoke-built) HDF5, or a differing test-row count within one option type
      all proceeded silently -- the same failure modes aggregate_results.
      validate_canonical already refuses for T1. They must hard-fail here too.

Run: pytest tests/test_ablation_guards.py  or  python tests/test_ablation_guards.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import eval_ablation as ab  # noqa: E402


def _rows(**over):
    """The four comparable runs, all provenance-clean. `over` patches one row."""
    rows = []
    for opt in ("call", "put"):
        for variant in ("per_query", "scalar_sigma"):
            rows.append({
                "option_type": opt, "model_variant": variant,
                "results_dir": f"train_model_v3/{opt}/results_{variant}",
                "seed": ab.CANONICAL_SEED, "dataset_version": "v5",
                "h5_sha256": "abc" if opt == "call" else "def",
                "canonical_dataset": True,
                "quote_filter": "static no-arbitrage midpoint",
                "n_test": 819342 if opt == "call" else 1380379,
            })
    rows[-1].update(over)
    return rows


# --------------------------------------------------------------- (a) smoke stem

def test_full_run_uses_canonical_stem():
    assert ab.output_stem("results", "v5", None).name == "ablation_scalar_sigma_v5"


def test_subsample_never_overwrites_canonical_stem():
    smoke = ab.output_stem("results", "v5", 20000)
    assert smoke.name == "ablation_scalar_sigma_v5_smoke"
    assert smoke != ab.output_stem("results", "v5", None)


# ------------------------------------------------------ (b) provenance enforced

def test_clean_rows_pass():
    ab.validate_provenance(_rows())          # must not raise


@pytest.mark.parametrize("patch, expect", [
    ({"seed": 43}, "OFF-SEED"),
    ({"dataset_version": "v4"}, "MIXED dataset_version"),
    ({"h5_sha256": "deadbeef"}, "MIXED put HDF5 sha256"),
    ({"quote_filter": "none"}, "MIXED quote_filter"),
    ({"canonical_dataset": False}, "NON-CANONICAL"),
    ({"n_test": 12345}, "MIXED put test-row count"),
])
def test_mismatched_provenance_hard_fails(patch, expect):
    with pytest.raises(SystemExit) as exc:
        ab.validate_provenance(_rows(**patch))
    assert expect in str(exc.value)


def test_missing_seed_is_not_silently_accepted():
    with pytest.raises(SystemExit):
        ab.validate_provenance(_rows(seed=None))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
