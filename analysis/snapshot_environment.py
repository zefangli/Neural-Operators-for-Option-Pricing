#!/usr/bin/env python
"""
Snapshot of the environment THIS repository is audited / reproduced in.

Read the header of the file it writes before citing anything from it: the
August 2026 training runs were executed in an environment that was never
snapshotted, and nothing here reconstructs it. This file documents the machine
that re-ran the evaluators, the baselines and the figure scripts, and that is
all it can be cited for.

    python analysis/snapshot_environment.py            # -> results/environment_snapshot_<today>.txt
    python analysis/snapshot_environment.py --out X    # explicit path

Text only. No JSON, nothing machine-parsed downstream -- if a number from here
reaches the manuscript it is quoted by hand and audited by
analysis/audit_manuscript_numbers.py like any other literal.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEADER = """\
ENVIRONMENT SNAPSHOT -- AUDIT / REPRODUCTION MACHINE
====================================================
Captured: {when}
By      : analysis/snapshot_environment.py

WHAT THIS IS. The Python/CUDA/driver environment in which this repository's
evaluators, baselines, diagnostics and figure scripts were re-run for the
publication pass.

WHAT THIS IS NOT. It is NOT the training environment. The v5 model checkpoints
were trained in August 2026 and that environment was never captured; package
versions, the CUDA runtime and the driver may all have moved since. This file
cannot be used to reconstruct it, and no statement of the form "the models were
trained with <version from this file>" is supportable. Cite it only as the
environment the reported evaluation numbers were produced in.
"""

# (import name, label). SciPy/polars/pandas are optional here on purpose --
# absence is a fact worth recording, not a crash.
PACKAGES = [("numpy", "NumPy"), ("h5py", "h5py"), ("matplotlib", "matplotlib"),
            ("polars", "polars"), ("pandas", "pandas"), ("scipy", "SciPy")]


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (p.stdout or "").strip() or (p.stderr or "").strip() or "(no output)"
    except Exception as exc:                        # not installed / not on PATH
        return "unavailable (%s: %s)" % (type(exc).__name__, exc)


def _version(module_name):
    try:
        mod = __import__(module_name)
    except Exception as exc:
        return "not installed (%s)" % type(exc).__name__
    return str(getattr(mod, "__version__", "installed, version attribute absent")) + (
        " [%s]" % getattr(mod, "__file__", "?"))


def torch_block():
    try:
        import torch
    except Exception as exc:
        return ["PyTorch                : not installed (%s)" % type(exc).__name__]
    cudnn = torch.backends.cudnn.version()
    lines = ["PyTorch                : %s" % torch.__version__,
             "torch.version.cuda     : %s" % torch.version.cuda,
             "cuDNN                  : %s" % (cudnn if cudnn is not None else "unavailable"),
             "torch.cuda.is_available: %s" % torch.cuda.is_available()]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append("CUDA device %d          : %s (cc %d.%d, %.1f GiB)"
                         % (i, p.name, p.major, p.minor, p.total_memory / 2 ** 30))
    return lines


def snapshot():
    out = [HEADER.format(when=time.strftime("%Y-%m-%d %H:%M:%S %z"))]
    out += ["", "--- interpreter -----------------------------------------------------",
            "Python                 : %s" % sys.version.replace("\n", " "),
            "Executable             : %s" % sys.executable,
            "Implementation         : %s" % platform.python_implementation(),
            "", "--- torch / gpu -----------------------------------------------------"]
    out += torch_block()
    out += ["", "nvidia-smi:", _run(["nvidia-smi"]),
            "", "--- packages --------------------------------------------------------"]
    for mod, label in PACKAGES:
        out.append("%-22s : %s" % (label, _version(mod)))
    out += ["", "--- operating system ------------------------------------------------",
            "Platform               : %s" % platform.platform(),
            "System / release       : %s %s" % (platform.system(), platform.release()),
            "Version                : %s" % platform.version(),
            "Machine / processor    : %s / %s" % (platform.machine(), platform.processor()),
            "", "--- pip list --format=freeze ----------------------------------------",
            _run([sys.executable, "-m", "pip", "list", "--format=freeze"]), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default results/environment_snapshot_<YYYY-MM-DD>.txt)")
    args = ap.parse_args()
    out = args.out or (PROJECT_ROOT / "results" /
                       ("environment_snapshot_%s.txt" % time.strftime("%Y-%m-%d")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(snapshot(), encoding="utf-8")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
