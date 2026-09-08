"""Unit tests for analysis/audit_manuscript_numbers.py.

No dependency on the real manuscript or the real artifacts: the end-to-end test
builds a tiny .tex plus a tiny JSON/CSV fixture in tmp_path. What is pinned here
is the extraction of every LaTeX numeric form the manuscript actually uses, the
rounding policy (half-even and half-up accepted, truncation rejected), the
exponent-aware comparison that makes `$10^{-6}$` a claim about 1e-6 rather than
about zero, the bound-claim mode, and the status/category assignment including
the non-zero exit on MISMATCH / UNMAPPED.

Run: pytest tests/test_audit_manuscript_numbers.py
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import audit_manuscript_numbers as A  # noqa: E402


def lits(text, tmp_path):
    p = tmp_path / "t.tex"
    p.write_text(text, encoding="utf-8")
    rows, skipped = A.extract(p)
    return rows, skipped


# ------------------------------------------------------------------ extraction

@pytest.mark.parametrize("src,literal,value", [
    ("3{,}297{,}722 rows", "3{,}297{,}722", 3297722.0),
    ("$-3.090190$ here", "-3.090190", -3.090190),
    (r"$1.84\times10^{-14}$", r"1.84\times10^{-14}", 1.84e-14),
    (r"rate $10^{-3}$", "10^{-3}", 1e-3),
    (r"removes 2.21\% of rows", "2.21", 2.21),
    ("covering 2015-02-02 onward", "2015-02-02", "2015-02-02"),
    ("Python & 3.11.15", "3.11.15", "3.11.15"),
])
def test_each_latex_numeric_form_is_extracted_once(src, literal, value, tmp_path):
    rows, _ = lits(src, tmp_path)
    assert [r["literal"] for r in rows] == [literal]
    assert rows[0]["value"] == value


def test_a_range_yields_two_positive_literals_not_a_negative_one(tmp_path):
    rows, _ = lits("scores 0.990--0.991 overall", tmp_path)
    assert [r["value"] for r in rows] == [0.990, 0.991]


def test_scientific_notation_is_one_literal_not_three(tmp_path):
    rows, _ = lits(r"is $2\times10^{-15}$ small", tmp_path)
    assert len(rows) == 1 and rows[0]["value"] == 2e-15


def test_comments_are_skipped_but_escaped_percent_is_kept(tmp_path):
    rows, _ = lits("visible 12 % hidden 99\nalso 3.5\\% kept", tmp_path)
    assert [r["value"] for r in rows] == [12.0, 3.5]


def test_math_superscripts_on_a_symbol_are_not_numbers(tmp_path):
    rows, _ = lits(r"$R^2(\text{price})$ is high", tmp_path)
    assert rows == []


def test_typesetting_parameters_are_skipped_and_counted(tmp_path):
    rows, skipped = lits("\\documentclass[11pt]{article}\n"
                         "\\begin{subfigure}[b]{0.48\\linewidth}\n"
                         "real number 7 here", tmp_path)
    assert [r["value"] for r in rows] == [7.0]
    assert skipped >= 2


def test_section_and_float_labels_are_attached(tmp_path):
    tex = ("\\section{Results}\n"
           "in text 1.5\n"
           "\\begin{table}[t]\n"
           "cell 2.5\n"
           "\\label{tab:t1}\n"
           "\\end{table}\n"
           "after 3.5\n")
    rows, _ = lits(tex, tmp_path)
    where = {r["value"]: r["section_or_table"] for r in rows}
    assert where[1.5] == "Results"
    assert where[2.5] == "tab:t1"          # label comes AFTER the cell
    assert where[3.5] == "Results"


# -------------------------------------------------------------------- rounding

@pytest.mark.parametrize("printed,source,dp,ok", [
    (0.9986, 0.9985863019, 4, True),            # ordinary round-half-anything
    (0.125, 0.1245, 3, True),                   # half-up
    (0.124, 0.1245, 3, True),                   # half-even -- both accepted
    (0.9999, 0.99999, 4, False),                # truncation is NOT accepted
    (2.206, 2.2060499, 3, True),
    (2.206, 2.2078, 3, False),                  # rounds to 2.208 either way
])
def test_rounding_acceptance_and_rejection(printed, source, dp, ok):
    assert A.rounds_to(printed, source, dp) is ok


def test_printed_decimals_counts_only_what_is_printed():
    assert A.printed_decimals("0.9986") == 4
    assert A.printed_decimals("42") == 0
    assert A.printed_decimals(r"1.84\times10^{-14}") == 2
    assert A.printed_decimals("10^{-6}") == 0


def test_power_of_ten_is_compared_in_its_own_exponent():
    """`$10^{-6}$` claims 1e-6. A source of 1.19e-5 is an order of magnitude out
    and must fail -- naive quantisation would round both to 0 and pass."""
    assert A.exponent_of("10^{-6}") == -6
    assert A.rounds_to(1e-6, 1.19e-5, 0, -6) is False
    assert A.rounds_to(1e-6, 1.2e-6, 0, -6) is True
    assert A.rounds_to(2e-14, 1.84e-14, 0, -14) is True


# ------------------------------------------------- category / status assignment

def row(value, literal=None):
    return {"value": value, "literal": literal if literal is not None else repr(value)}


def test_unmapped_literal_gets_the_unmapped_status():
    cat, status, _f, _k, _v = A.classify(row(1.0), None)
    assert (cat, status) == ("", "UNMAPPED")


def test_a_category_outside_the_allowed_six_is_a_mismatch():
    _c, status, _f, key, _v = A.classify(row(1.0), {"category": "vibes",
                                                    "source_value": 1.0})
    assert status == "MISMATCH" and "not one of the six" in key


def test_unsupported_category_short_circuits_to_unsupported():
    cat, status, _f, key, val = A.classify(
        row(1.0), {"category": "unsupported", "note": "no artifact"})
    assert (cat, status, val) == ("unsupported", "UNSUPPORTED", "")
    assert key == "no artifact"


def test_exact_rounded_and_derived_statuses():
    e = {"category": "experiment output", "source_value": 0.5}
    assert A.classify(row(0.5), e)[1] == "EXACT"
    e2 = {"category": "experiment output", "source_value": 0.5004}
    assert A.classify(row(0.5, "0.500"), e2)[1] == "ROUNDED-OK"
    d = {"category": "derived statistic", "derivation": "0.25 + 0.25"}
    assert A.classify(row(0.5), d)[1] == "DERIVED"
    d2 = {"category": "derived statistic", "derivation": "0.5004"}
    assert A.classify(row(0.5, "0.500"), d2)[1] == "DERIVED"


def test_a_disagreeing_source_is_a_mismatch():
    e = {"category": "experiment output", "source_value": 0.6}
    assert A.classify(row(0.5, "0.5"), e)[1] == "MISMATCH"


def test_bound_claims_are_not_rescued_by_rounding():
    """'sits within 5e-6' when the value is 5.487e-6 rounds fine but is false."""
    e = {"category": "derived statistic", "source_value": 5.487e-6,
         "comparison": "upper_bound"}
    assert A.classify(row(5e-6, r"5\times10^{-6}"), e)[1] == "MISMATCH"
    e["source_value"] = 4.9e-6
    assert A.classify(row(5e-6, r"5\times10^{-6}"), e)[1] == "EXACT"


def test_a_broken_map_entry_is_reported_as_a_mismatch_not_an_exception():
    e = {"category": "experiment output", "source_file": "nope.json", "source": "a.b"}
    cat, status, _f, key, _v = A.classify(row(1.0), e)
    assert status == "MISMATCH" and "failed to resolve" in key


def test_string_valued_literals_compare_exactly():
    e = {"category": "dataset/config/provenance", "source_value": "3.11.15"}
    assert A.classify({"value": "3.11.15", "literal": "3.11.15"}, e)[1] == "EXACT"
    e["source_value"] = "3.12.0"
    assert A.classify({"value": "3.11.15", "literal": "3.11.15"}, e)[1] == "MISMATCH"


# ------------------------------------------------------------------ end to end

FIXTURE_TEX = """\
\\section{Results}
The model reaches 0.999977 on price and 0.99 on log.
Rows removed: 63{,}101, i.e. 1.878\\%.
It sits within $5\\times10^{-6}$ of the best baseline.
An unmapped 12345 appears here.
The caption says architecture-A/B models.
"""


def _fixture(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"r2_price": 0.9999769212649592, "r2_log_filtered": 0.9910439390352741}))
    (tmp_path / "counts.json").write_text(json.dumps({"v4": 3360823, "v5": 3297722}))
    tex = tmp_path / "m.tex"
    tex.write_text(FIXTURE_TEX, encoding="utf-8")
    m = {
        "literals": {
            "0.999977": {"category": "experiment output",
                         "source_file": "metrics.json", "source": "r2_price"},
            "0.99": {"category": "experiment output",
                     "source_file": "metrics.json", "source": "r2_log_filtered"},
            "63101": {"category": "derived statistic", "source_file": "counts.json",
                      "derivation": "J('counts.json','v4') - J('counts.json','v5')"},
            "1.878": {"category": "derived statistic", "source_file": "counts.json",
                      "derivation": "100.0*(J('counts.json','v4') - J('counts.json','v5'))"
                                    "/J('counts.json','v4')"},
            "5\\times10^{-6}": {"category": "derived statistic", "comparison": "upper_bound",
                                "source_file": "counts.json", "source_value": 5.487e-6},
        },
        "by_line": {},
        "text_checks": [{"line": 6, "literal": "architecture-A/B",
                         "expected": "architecture-A", "category": "unsupported",
                         "note": "caption overstates which models the figure shows"}],
    }
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    return tex, mp


def test_end_to_end_on_a_synthetic_tex_and_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "PROJECT_ROOT", tmp_path)
    A._JSON.clear(), A._CSV.clear(), A._TEXT.clear()
    tex, mp = _fixture(tmp_path)
    rows, _skipped = A.audit(tex, mp)
    by = {r["literal"]: r for r in rows}

    assert by["0.999977"]["status"] == "ROUNDED-OK"
    assert by["0.999977"]["category"] == "experiment output"
    assert by["0.99"]["status"] == "ROUNDED-OK"
    assert by["63{,}101"]["status"] == "DERIVED"       # keyed by the PRINTED form
    assert by["63{,}101"]["value"] == "63101.0"
    assert by["1.878"]["status"] == "DERIVED"
    assert by["5\\times10^{-6}"]["status"] == "MISMATCH"  # bound violated
    assert by["12345"]["status"] == "UNMAPPED"
    assert by["architecture-A/B"]["status"] == "MISMATCH"
    assert by["architecture-A/B"]["category"] == "unsupported"
    assert [r["line"] for r in rows] == sorted(r["line"] for r in rows)


def test_main_exits_nonzero_when_anything_is_unmapped_or_mismatched(tmp_path, monkeypatch,
                                                                    capsys):
    monkeypatch.setattr(A, "PROJECT_ROOT", tmp_path)
    A._JSON.clear(), A._CSV.clear(), A._TEXT.clear()
    tex, mp = _fixture(tmp_path)
    out = tmp_path / "prov.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--tex", str(tex), "--map", str(mp),
                                      "--out", str(out)])
    assert A.main() == 1
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0]
    for col in ("line", "section_or_table", "literal", "value", "category",
                "source_file", "source_key_or_derivation", "source_value",
                "mapping_scope", "status"):
        assert col in header
    assert out.with_suffix(".md").exists()          # the companion caveat page


# ----------------------------------------------------------- mapping scope

SCOPE_TEX = """\
\\section{S}
the metric is 0.5 here
and 0.5 again over there
and 0.75 only once
"""


def _scope_case(tmp_path, override):
    tex = tmp_path / "scope.tex"
    tex.write_text(SCOPE_TEX, encoding="utf-8")
    mp = tmp_path / "scope_map.json"
    mp.write_text(json.dumps({
        "literals": {"0.5": {"category": "experiment output", "source_value": 0.5,
                             "note": "the global note"},
                     "0.75": {"category": "experiment output", "source_value": 0.75}},
        "by_line": {"3:0.5": override},
        "text_checks": [],
    }), encoding="utf-8")
    rows, _ = A.audit(tex, mp)
    return {r["line"]: r for r in rows}


def test_mapping_scope_is_line_for_a_by_line_entry_and_global_otherwise(tmp_path):
    by_line = _scope_case(tmp_path, {"category": "experiment output",
                                     "source_value": 0.5, "note": "its own source"})
    assert by_line[2]["mapping_scope"] == "global"
    assert by_line[3]["mapping_scope"] == "line"
    assert by_line[4]["mapping_scope"] == "global"
    assert "its own source" in by_line[3]["source_key_or_derivation"]


def test_a_by_line_string_is_a_review_note_that_keeps_the_global_source(tmp_path):
    """The cheap override: the form's own entry still resolves the value, and the
    string records what THIS occurrence was confirmed to name."""
    by_line = _scope_case(tmp_path, "reviewed: the second occurrence, same metric")
    assert by_line[3]["mapping_scope"] == "line"
    assert by_line[3]["status"] == "EXACT"          # global entry supplied the source
    key = by_line[3]["source_key_or_derivation"]
    assert "the global note" in key and "reviewed: the second occurrence" in key


def test_repeated_global_forms_reports_only_multi_line_forms_with_global_rows(tmp_path):
    rows = list(_scope_case(tmp_path, "reviewed").values())
    forms, n_global = A.repeated_global_forms(rows)
    assert forms == [("0.5", 1, [2])]               # 0.75 occurs once: not reported
    assert n_global == 1


def test_main_exits_zero_when_everything_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "PROJECT_ROOT", tmp_path)
    A._JSON.clear(), A._CSV.clear(), A._TEXT.clear()
    tex = tmp_path / "clean.tex"
    tex.write_text("\\section{S}\nvalue 0.5 only.\n", encoding="utf-8")
    mp = tmp_path / "clean_map.json"
    mp.write_text(json.dumps({"literals": {"0.5": {"category": "theoretical constant",
                                                   "source_value": 0.5}},
                              "by_line": {}, "text_checks": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--tex", str(tex), "--map", str(mp),
                                      "--out", str(tmp_path / "o.csv")])
    assert A.main() == 0
