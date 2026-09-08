#!/usr/bin/env python
"""
Automated consistency guardrail for the numeric literals printed in paper/main.tex.

WHAT IT IS. One maintained script, replacing the throwaway scripts of the
pre-submission audit. It extracts each numeric literal from the manuscript,
looks up the artifact (or the explicit derivation over artifacts) that literal
is mapped to, re-reads that artifact, and checks that the printed digits agree
with it at the printed precision.

    python analysis/audit_manuscript_numbers.py
    python analysis/audit_manuscript_numbers.py --dump-unmapped   # map maintenance
    python analysis/audit_manuscript_numbers.py --tex x.tex --map m.json --out o.csv

Writes results/manuscript_number_provenance.csv plus a short
results/manuscript_number_provenance.md, and prints a summary by category and
status. **Exits non-zero if any MISMATCH, UNMAPPED or UNSUPPORTED row exists,
or if any `by_line` override in the map matched no occurrence.** That is the
point: a manuscript number no artifact supports must break something, and so
must a per-occurrence override that has silently stopped applying.

WHAT IT IS NOT. Passing is not a proof that every occurrence has the correct
SEMANTIC source. The check is value equality against a mapped artifact, and the
map is keyed by the literal form: one entry in `literals` covers every
occurrence of that form unless a `by_line` override exists for the occurrence.
So a repeated form -- `0.991044` in three places, `10` in eighteen -- is
verified against one source for all of them, and a second occurrence that
happened to mean something else with the same digits would still pass. The
`mapping_scope` column records which case a row is in: `line` where a `by_line`
override established this occurrence individually, `global` where the form's
single `literals` entry was applied. The summary reports how many `global` rows
belong to forms that occur on more than one line, and lists those forms; those
are exactly the rows whose semantics rest on the form-level mapping rather than
on a per-occurrence one.

BY_LINE OVERRIDES take two shapes. A full entry (a JSON object, with its own
category/source/derivation) is used when the occurrence has a DIFFERENT source
from the form's global entry. A plain string is a review note: the global entry
still supplies the source, and the note records what the occurrence was
confirmed to name. Both count as `line` scope. A `by_line` key is
`<line>:<normalised literal>`, so editing main.tex moves occurrences out from
under the keys: every declared override must be consumed by an actual
occurrence at that line with that literal, or the run FAILS and names the stale
keys instead of quietly falling back to the form's global entry.

CATEGORIES (exactly six; assigned in analysis/manuscript_number_map.json):
    experiment output           a metric or count an evaluator / training run wrote
    derived statistic           computed here from artifact values; formula recorded
    dataset/config/provenance   dataset counts, dates, hyper-parameters, versions
    theoretical constant        fixed by the mathematics or by the sentence's own
                                definition (a limit T -> 0, a grid dimension, a
                                tolerance the text itself sets)
    literature value            attributed to a cited work
    unsupported                 nothing in this repository backs it

STATUSES:
    EXACT        printed digits equal the source value exactly
    ROUNDED-OK   the source value rounds to the printed digits (half-even OR
                 half-up accepted; TRUNCATION IS NOT -- a figure that only
                 matches under truncation is reported MISMATCH, because it
                 claims a precision the artifact does not support)
    DERIVED      a derivation formula over artifact values reproduces the digits
    UNSUPPORTED  category `unsupported`: no source exists. Reported, never hidden
    MISMATCH     a source exists and disagrees at printed precision, or a stated
                 bound is violated by the value it bounds
    UNMAPPED     no entry in the map for this literal

WHAT IS DELIBERATELY NOT AUDITED. LaTeX typesetting parameters are not
manuscript claims and are skipped, with the count reported: option lists of
\\documentclass / \\usepackage / \\usetikzlibrary / \\setlength, TeX length
literals (`11pt`, `1in`), and float widths (`0.48\\linewidth`). Math
superscripts on a symbol (the `2` of `$R^2$`) are likewise not numeric
literals; only `10^{...}` powers are.

COMPARISON RULES. A power-of-ten literal is compared IN ITS OWN EXPONENT: the
source value is divided by 10^exponent and rounded to the mantissa's printed
decimals, so `$10^{-6}$` is checked against 1 (not 0) in units of 1e-6 and a
source of 1.19e-5 fails. A map entry may set `"comparison": "upper_bound"` for
a sentence that states a BOUND ("sits within 5e-6"): the source must be less
than or equal to the printed value, and rounding does not rescue it.

OPEN ITEMS are never suppressed. A number no artifact supports is reported as
MISMATCH / UNMAPPED / UNSUPPORTED and the non-zero exit it causes is the
expected state until the manuscript or the map is fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = PROJECT_ROOT / "paper" / "main.tex"
DEFAULT_MAP = Path(__file__).resolve().parent / "manuscript_number_map.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "manuscript_number_provenance.csv"

CATEGORIES = ("experiment output", "derived statistic", "dataset/config/provenance",
              "theoretical constant", "literature value", "unsupported")

# Priority-ordered alternation: leftmost match wins and, at one position, the
# first listed alternative wins -- so `1.84\times10^{-14}` is ONE literal, not
# three, and `2015-02-02` is one date, not three integers. The `num` lookbehind
# excludes `^` so the 2 of `$R^2$` is not mistaken for a number.
NUM_RE = re.compile(r"""
    (?P<date>(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-]))
  | (?P<version>(?<![A-Za-z0-9_.^])\d+\.\d+\.\d+(?![\d.]))
  | (?P<sci>(?<![A-Za-z0-9_.^])-?\d+(?:\.\d+)?\s*\\times\s*10\^\{?-?\d+\}?)
  | (?P<pow>(?<![A-Za-z0-9_.^\d])10\^\{?-?\d+\}?)
  | (?P<num>(?<![A-Za-z0-9_.^])-?\d[\d{},.]*\d|(?<![A-Za-z0-9_.^])\d(?![\d.]))
""", re.VERBOSE)
SCI_PARTS = re.compile(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?")
POW_PARTS = re.compile(r"10\^\{?(-?\d+)\}?")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
COMMENT_RE = re.compile(r"(?<!\\)%")
SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")
LABEL_RE = re.compile(r"\\label\{((?:tab|fig|sec|eq|app):[^}]*)\}")

# --- typesetting, not manuscript claims (see docstring) ---
TYPESET_LINE = re.compile(r"^\s*\\(documentclass|usepackage|usetikzlibrary|setlength"
                          r"|captionsetup|geometry|hypersetup|bibliographystyle)\b")
TYPESET_TOKEN = re.compile(r"\d+(?:\.\d+)?(?:pt|in|cm|mm|em|ex|bp|sp)(?![A-Za-z])"
                           r"|\d*\.?\d+\s*\\(?:line|text|column)width")


# --------------------------------------------------------------- extraction

def strip_comment(line):
    r"""Everything before the first unescaped `%`; `\%` is a printed percent."""
    m = COMMENT_RE.search(line)
    return line[:m.start()] if m else line


def literal_value(text):
    r"""Value of one extracted literal: float, or str for a date / version.

    Handles `3{,}297{,}722`, `-3.090190`, `1.84\times10^{-14}`, `10^{-3}`,
    `2015-02-02` and `3.11.15`. Returns None if the text is not one of these.
    """
    t = text.strip()
    if DATE_RE.fullmatch(t) or VERSION_RE.fullmatch(t):
        return t
    m = SCI_PARTS.fullmatch(t)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    m = POW_PARTS.fullmatch(t)
    if m:
        return 10.0 ** int(m.group(1))
    clean = t.replace("{,}", "").replace(",", "").rstrip(".")
    try:
        return float(clean)
    except ValueError:
        return None


def normalise(text):
    """Map-lookup key: thousands separators removed, whitespace collapsed."""
    t = text.strip()
    if SCI_PARTS.fullmatch(t) or POW_PARTS.fullmatch(t):
        return re.sub(r"\s+", "", t)
    if DATE_RE.fullmatch(t) or VERSION_RE.fullmatch(t):
        return t
    return t.replace("{,}", "").replace(",", "").rstrip(".")


def float_spans(lines):
    r"""{line number: label} for every line inside a labelled table/figure.

    A number inside `\begin{table} ... \label{tab:t1} ... \end{table}` belongs
    to that table even though the label line comes after it.
    """
    spans, stack = [], []
    for i, raw in enumerate(lines, 1):
        s = strip_comment(raw)
        for env in ("table*", "figure*", "table", "figure"):
            if r"\begin{%s}" % env in s:
                stack.append([i, None])
                break
        m = LABEL_RE.search(s)
        if m and stack:
            stack[-1][1] = stack[-1][1] or m.group(1)
        for env in ("table*", "figure*", "table", "figure"):
            if r"\end{%s}" % env in s and stack:
                start, label = stack.pop()
                spans.append((start, i, label))
                break
    out = {}
    for start, end, label in spans:
        if label:
            for ln in range(start, end + 1):
                out[ln] = label
    return out


def extract(tex_path):
    """-> ([{line, section_or_table, literal, norm, value, context}], n_skipped)."""
    lines = Path(tex_path).read_text(encoding="utf-8").splitlines()
    floats = float_spans(lines)
    section, out, n_skipped = "(preamble)", [], 0
    for i, raw in enumerate(lines, 1):
        s = strip_comment(raw)
        m = SECTION_RE.search(s)
        if m:
            section = m.group(1)
        if not s.strip():
            continue
        if TYPESET_LINE.match(s):
            n_skipped += len(NUM_RE.findall(s))
            continue
        s, n = TYPESET_TOKEN.subn(" ", s)
        n_skipped += n
        # `a--b` is a range, not a minus sign: mask it so `b` stays positive.
        scan = s.replace("--", "\u2013\u2013")
        for mm in NUM_RE.finditer(scan):
            text = mm.group(0).replace("\u2013\u2013", "--")
            val = literal_value(text)
            if val is None:
                continue
            out.append({"line": i, "section_or_table": floats.get(i, section),
                        "literal": text, "norm": normalise(text), "value": val,
                        "context": s.strip()})
    return out, n_skipped


# --------------------------------------------------------- artifact readers

_JSON, _CSV, _TEXT = {}, {}, {}


def _resolve_path(rel):
    """Project-relative path; a `*` glob picks the last match (dated files)."""
    if "*" in rel:
        hits = sorted(PROJECT_ROOT.glob(rel))
        if not hits:
            raise FileNotFoundError("no file matches %s" % rel)
        return hits[-1]
    return PROJECT_ROOT / rel


def J(rel, path):
    """Dotted lookup into a JSON artifact.

    A path element indexes a dict, or a list by integer, or SELECTS one record
    of a list of dicts with `field=value` (join several with `&`).
    """
    if rel not in _JSON:
        _JSON[rel] = json.loads(_resolve_path(rel).read_text(encoding="utf-8"))
    cur = _JSON[rel]
    for part in str(path).split("."):
        if isinstance(cur, list):
            if part.isdigit():
                cur = cur[int(part)]
                continue
            hits = list(cur)
            for cond in part.split("&"):
                k, v = cond.split("=", 1)
                hits = [r for r in hits if str(r.get(k)) == v]
            if len(hits) != 1:
                raise KeyError("%s: selector %r matched %d records"
                               % (rel, part, len(hits)))
            cur = hits[0]
            continue
        cur = cur[part]
    return cur


def C(rel, selector, field):
    """One CSV cell. `selector` is `k=v` or `k=v,k2=v2`; must match exactly one row."""
    if rel not in _CSV:
        with open(_resolve_path(rel), newline="", encoding="utf-8") as f:
            _CSV[rel] = list(csv.DictReader(f))
    rows = _CSV[rel]
    for cond in selector.split(","):
        k, v = cond.split("=", 1)
        rows = [r for r in rows if r.get(k) == v]
    if len(rows) != 1:
        raise KeyError("%s: selector %r matched %d rows" % (rel, selector, len(rows)))
    return rows[0][field]


def R(rel, pattern, group=1):
    """First regex capture in a text artifact (markdown table, snapshot, source)."""
    if rel not in _TEXT:
        _TEXT[rel] = _resolve_path(rel).read_text(encoding="utf-8", errors="replace")
    m = re.search(pattern, _TEXT[rel])
    if not m:
        raise KeyError("%s: no match for %r" % (rel, pattern))
    return m.group(group)


def EPOCHS(rel):
    """[(epoch, train_loss, val_loss)] parsed from a run's `loss_history.txt`."""
    out = []
    for line in _resolve_path(rel).read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split("\t")
        if len(p) == 3 and p[0].isdigit():
            out.append((int(p[0]), float(p[1]), float(p[2])))
    if not out:
        raise KeyError("%s: no epoch rows" % rel)
    return out


SAFE_NS = {"J": J, "C": C, "R": R, "EPOCHS": EPOCHS,
           "abs": abs, "min": min, "max": max, "sum": sum,
           "len": len, "float": float, "int": int, "round": round, "str": str,
           "sorted": sorted, "exp": math.exp, "log": math.log, "log1p": math.log1p,
           "sqrt": math.sqrt, "__builtins__": {}}


def resolve(entry):
    """-> (source_value, source_key_or_derivation). Raises if the entry is broken."""
    if "derivation" in entry:
        return eval(entry["derivation"], dict(SAFE_NS)), entry["derivation"]  # noqa: S307
    if "source_regex" in entry:
        return R(entry["source_file"], entry["source_regex"]), \
            "regex %s" % entry["source_regex"]
    if "source" in entry:
        rel = entry["source_file"]
        if rel.endswith(".csv"):
            sel, field = entry["source"].rsplit(":", 1)
            return C(rel, sel, field), entry["source"]
        return J(rel, entry["source"]), entry["source"]
    if "source_value" in entry:
        return entry["source_value"], entry.get("note", "definitional")
    raise KeyError("map entry has no source, derivation, source_regex or source_value")


# ------------------------------------------------------------- comparison

def exponent_of(literal):
    """The printed power-of-ten exponent (0 for a plain decimal literal)."""
    t = normalise(literal)
    m = SCI_PARTS.fullmatch(t)
    if m:
        return int(m.group(2))
    m = POW_PARTS.fullmatch(t)
    if m:
        return int(m.group(1))
    return 0


def printed_decimals(literal):
    """Decimals actually printed, so 0.9986 is checked to 4 places and not to 15."""
    t = normalise(literal)
    m = SCI_PARTS.fullmatch(t)
    if m:
        t = m.group(1)
    elif POW_PARTS.fullmatch(t):
        return 0
    return len(t.split(".")[1]) if "." in t else 0


def rounds_to(printed_value, source_value, decimals, exponent=0):
    """Does `source_value` round to `printed_value` at the printed precision?

    Both are first expressed in units of 10**exponent, so a power-of-ten literal
    is judged on its own mantissa: `10^{-6}` is 1 (in units of 1e-6) and a source
    of 1.19e-5 is 11.9, which does not round to 1. Half-even and half-up are both
    accepted; truncation is not.
    """
    scale = Decimal(10) ** exponent
    d = (Decimal(repr(float(source_value))) / scale)
    p = (Decimal(repr(float(printed_value))) / scale)
    quant = Decimal(1).scaleb(-decimals)
    p = p.quantize(quant, rounding=ROUND_HALF_EVEN)
    return any(d.quantize(quant, rounding=mode) == p
               for mode in (ROUND_HALF_EVEN, ROUND_HALF_UP))


def classify(row, entry):
    """-> (category, status, source_file, source_key_or_derivation, source_value)."""
    if entry is None:
        return "", "UNMAPPED", "", "", ""
    cat = entry.get("category", "")
    if cat not in CATEGORIES:
        return cat, "MISMATCH", entry.get("source_file", ""), \
            "category %r is not one of the six allowed" % cat, ""
    if cat == "unsupported":
        return cat, "UNSUPPORTED", entry.get("source_file", ""), \
            entry.get("note", "no artifact in this repository backs this number"), ""
    try:
        src_val, src_key = resolve(entry)
    except Exception as exc:                      # a broken map entry IS a failure
        return cat, "MISMATCH", entry.get("source_file", ""), \
            "map entry failed to resolve: %s: %s" % (type(exc).__name__, exc), ""
    src_file = entry.get("source_file", "")
    note = entry.get("note")
    if note:
        src_key = "%s  [%s]" % (src_key, note)
    if src_val is None:
        return cat, "MISMATCH", src_file, src_key, ""

    if isinstance(row["value"], str) or isinstance(src_val, str) and \
            not _is_number(src_val):
        ok = str(src_val).strip() == str(row["value"]).strip()
        return cat, "EXACT" if ok else "MISMATCH", src_file, src_key, str(src_val)

    src_num = float(src_val)
    if entry.get("comparison") == "upper_bound":
        # "sits within X" is a bound claim: rounding cannot rescue a violation.
        ok = src_num <= row["value"]
        return cat, "EXACT" if ok else "MISMATCH", src_file, \
            src_key + "  [bound claim: source must be <= the printed value]", repr(src_num)

    dp, ex = printed_decimals(row["literal"]), exponent_of(row["literal"])
    if src_num == row["value"]:
        status = "DERIVED" if "derivation" in entry else "EXACT"
    elif rounds_to(row["value"], src_num, dp, ex):
        status = "DERIVED" if "derivation" in entry else "ROUNDED-OK"
    else:
        status = "MISMATCH"
    return cat, status, src_file, src_key, repr(src_num)


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------- text checks

def run_text_checks(checks, tex_path):
    """Non-numeric claims the audit must still flag (a caption that overstates
    which models a figure shows, say). Each check is {line, literal, expected,
    category, source_file, note}: if `literal` is present on `line` the check
    FAILS, because the map records it as the wrong text."""
    lines = Path(tex_path).read_text(encoding="utf-8").splitlines()
    out = []
    for chk in checks:
        ln = int(chk["line"])
        present = chk["literal"] in lines[ln - 1] if 0 < ln <= len(lines) else False
        out.append(OrderedDict([
            ("line", ln), ("section_or_table", chk.get("section_or_table", "")),
            ("literal", chk["literal"]), ("value", ""),
            ("category", chk.get("category", "unsupported")),
            ("source_file", chk.get("source_file", "")),
            ("source_key_or_derivation", chk.get("note", "")),
            ("source_value", chk.get("expected", "")),
            ("mapping_scope", "line"),          # text checks are keyed by line
            ("status", chk.get("status", "MISMATCH") if present else "EXACT"),
            ("context", (lines[ln - 1].strip() if 0 < ln <= len(lines) else "")[:200]),
        ]))
    return out


# ------------------------------------------------------------------- driver

def audit(tex_path, map_path):
    """-> (rows, n_skipped, stale_by_line_keys). Stale = declared, never consumed."""
    the_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
    entries, overrides = the_map["literals"], the_map.get("by_line", {})
    rows, n_skipped = extract(tex_path)
    out, used = [], set()
    for r in rows:
        key = "%d:%s" % (r["line"], r["norm"])
        override = overrides.get(key)
        if override is not None:
            used.add(key)
        review_note = None
        if isinstance(override, str):
            # A by_line string is a review note; the global entry keeps the source.
            override, review_note = None, override
        entry = override or entries.get(r["norm"])
        if review_note and entry is not None:
            entry = dict(entry, note="%s  [by-line review: %s]"
                         % (entry.get("note", "").strip(), review_note))
        cat, status, src_file, src_key, src_val = classify(r, entry)
        out.append(OrderedDict([
            ("line", r["line"]), ("section_or_table", r["section_or_table"]),
            ("literal", r["literal"]),
            ("value", r["value"] if isinstance(r["value"], str) else repr(r["value"])),
            ("category", cat), ("source_file", src_file),
            ("source_key_or_derivation", src_key), ("source_value", src_val),
            ("mapping_scope",
             "line" if (override is not None or review_note is not None) else "global"),
            ("status", status), ("context", r["context"][:200]),
        ]))
    out += run_text_checks(the_map.get("text_checks", []), tex_path)
    out.sort(key=lambda r: (r["line"], r["literal"]))
    return out, n_skipped, sorted(set(overrides) - used)


def repeated_global_forms(rows):
    """-> ([(form, n_global_rows, [lines])], n_global_rows_total).

    Literal forms that appear on more than one line AND still have at least one
    `global`-scoped row. Those rows are checked for VALUE against the form's one
    mapped source; nothing establishes that the occurrence means that source.
    """
    lines_of, globals_of = {}, {}
    for r in rows:
        form = normalise(r["literal"])
        lines_of.setdefault(form, set()).add(r["line"])
        if r.get("mapping_scope") == "global":
            globals_of.setdefault(form, []).append(r["line"])
    out = [(f, len(ls), sorted(set(ls)))
           for f, ls in sorted(globals_of.items()) if len(lines_of[f]) > 1]
    return out, sum(n for _f, n, _l in out)


def summarise(rows, n_skipped, stale):
    by_cat = Counter(r["category"] or "(unmapped)" for r in rows)
    by_status = Counter(r["status"] for r in rows)
    print("\n%d audited literals in %d distinct forms (%d typesetting tokens skipped)\n"
          % (len(rows), len({r["literal"] for r in rows}), n_skipped))
    print("BY CATEGORY")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print("  %-28s %5d" % (k, v))
    print("\nBY STATUS")
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print("  %-28s %5d" % (k, v))
    forms, n_global = repeated_global_forms(rows)
    print("\nMAPPING SCOPE")
    for k, v in sorted(Counter(r.get("mapping_scope", "") for r in rows).items()):
        print("  %-28s %5d" % (k, v))
    print("  %-28s %5d" % ("stale by_line overrides", len(stale)))
    print("  %d row(s) in %d repeated literal form(s) are global-scoped: their value is\n"
          "  checked, their per-occurrence semantic source is not individually established."
          % (n_global, len(forms)))
    for form, n, lines in forms:
        print("    %-22s x%-3d lines %s" % (form, n, lines[:12]))

    if stale:
        print("\nSTALE by_line OVERRIDES (%d): declared in the map but consumed by no"
              " occurrence -- the manuscript line moved or the literal changed, and the"
              " rows that should carry them fell back to the form's global entry."
              % len(stale))
        for k in stale:
            print("    %s" % k)

    bad = [r for r in rows if r["status"] in ("MISMATCH", "UNMAPPED", "UNSUPPORTED")]
    if bad:
        print("\nFLAGGED ROWS (%d)" % len(bad))
        for r in bad:
            print("  L%-5d %-11s %-20s %-26s src=%s"
                  % (r["line"], r["status"], r["literal"], r["category"] or "-",
                     str(r["source_value"])[:24]))
            print("        %s" % (r["source_key_or_derivation"] or r["context"])[:150])
    return by_cat, by_status


MD_TEMPLATE = """\
# What `manuscript_number_provenance.csv` is, and what it is not

Generated by `analysis/audit_manuscript_numbers.py`. Do not edit by hand.

## What it is

An **automated consistency guardrail**. Every numeric literal printed in
`paper/main.tex` is extracted, looked up in `analysis/manuscript_number_map.json`,
resolved to an artifact in this repository (or to an explicit derivation over
artifacts), and compared against the printed digits *at the printed precision*.
Rounding is accepted, truncation is not; a literal written as a bound is checked
as a bound. The script exits non-zero if any row is MISMATCH, UNMAPPED or
UNSUPPORTED, so a manuscript number that no artifact supports fails the run, and
also if any `by_line` override in the map matched no occurrence, so an override
cannot silently stop applying when manuscript lines shift.

Current run: **{n_rows} rows**, {status_line}.

## What it is not

It is **not** by itself a complete independent proof that every occurrence has
the correct semantic source. Two limits are structural:

1. The map is keyed by the **literal form**, not by the occurrence. One entry in
   `literals` covers every occurrence of that form unless a `by_line` override
   exists. A repeated form is therefore verified against a single source for all
   of its occurrences, and a second occurrence that happened to mean something
   else with the same digits would still pass.
2. Agreement with an artifact says nothing about whether the artifact itself is
   right, or whether the sentence around the number reads it correctly.

The `mapping_scope` column separates the two cases: `line` where a `by_line`
override established this occurrence individually, `global` where the form's one
`literals` entry was applied.

## Global-scoped repeated forms in this run

**{n_global} row(s) across {n_forms} literal form(s)** are `global`-scoped and
occur on more than one line. Their values are checked; their per-occurrence
semantic source is not individually established.

These are left global deliberately: they are theoretical constants and
config/provenance values -- grid dimensions, seeds, split shares, tolerances,
batch sizes, dataset dates and row counts. Each such occurrence has its value
checked against the form's mapped source; what the occurrence means in its own
sentence is not established by this run and remains unverified. Every repeated
form whose category is *experiment output* or *derived statistic*, where two
occurrences plausibly could mean different quantities, carries a `by_line`
review note instead.

{form_table}
"""


def write_markdown(path, rows, by_status):
    forms, n_global = repeated_global_forms(rows)
    if forms:
        table = ["| literal form | global rows | lines |", "| --- | --- | --- |"]
        table += ["| `%s` | %d | %s |" % (f, n, ", ".join(str(x) for x in lines))
                  for f, n, lines in forms]
    else:
        table = ["_None: every repeated form is covered by a `by_line` override._"]
    path.write_text(MD_TEMPLATE.format(
        n_rows=len(rows), n_global=n_global, n_forms=len(forms),
        status_line=", ".join("%d %s" % (v, k) for k, v in sorted(by_status.items())),
        form_table="\n".join(table)), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    ap.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dump-unmapped", action="store_true",
                    help="print every unmapped literal with context and exit "
                         "(map maintenance; writes no CSV)")
    args = ap.parse_args()

    rows, n_skipped, stale = audit(args.tex, args.map_path)
    if args.dump_unmapped:
        seen = {}
        for r in rows:
            if r["status"] == "UNMAPPED":
                seen.setdefault(r["literal"], []).append(r)
        print("%d distinct unmapped literals" % len(seen))
        for lit, rs in seen.items():
            print("\n%-18s x%d  lines %s" % (lit, len(rs), [x["line"] for x in rs][:10]))
            for x in rs[:2]:
                print("      L%d [%s] %s"
                      % (x["line"], x["section_or_table"], x["context"][:160]))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % args.out)
    _cat, by_status = summarise(rows, n_skipped, stale)
    print("wrote %s" % write_markdown(args.out.with_suffix(".md"), rows, by_status))
    # `unsupported` is also a CATEGORY carried by text-check guard rows that pass
    # with status EXACT; only the STATUS UNSUPPORTED means an unbacked number.
    n_bad = by_status["MISMATCH"] + by_status["UNMAPPED"] + by_status["UNSUPPORTED"]
    if n_bad:
        print("\nEXIT 1: %d MISMATCH + %d UNMAPPED + %d UNSUPPORTED. Intended "
              "behaviour -- a manuscript number no artifact supports must fail the audit."
              % (by_status["MISMATCH"], by_status["UNMAPPED"], by_status["UNSUPPORTED"]))
    if stale:
        print("\nEXIT 1: %d by_line override(s) matched no occurrence (listed "
              "above). Fix the line number or delete the override." % len(stale))
    return 1 if (n_bad or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
