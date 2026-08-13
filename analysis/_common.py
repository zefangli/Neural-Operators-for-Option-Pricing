"""
Shared building blocks for the paper's analysis / publication artifacts.

This module is intentionally SELF-CONTAINED (per CLAUDE.md): it re-implements the
HDF5 loader, the model stack (both readouts), the analytical Black-Scholes pricer,
and the canonical metric schema rather than importing the train_*.py scripts. The
train scripts are import-safe (guarded by `if __name__ == "__main__"`), but they
each redefine identically-named classes, so we keep one clean copy here.

Everything here mirrors `train_model_v3/*/train_vol_surface{,_don}.py` verbatim so
that `load_state_dict(..., strict=True)` succeeds against the saved `best_model.pth`
and the metrics reproduce each run's `loss_history.txt` numbers exactly.

Verification gate (see analysis/PROGRESS_paper_artifacts.md): running
`eval_to_json.py` on call/results_vol_surface must reproduce R2(price)=0.999851
and R2(log,T>1day)=0.992481 within rounding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
T_MIN_YEARS = 1.0 / 365.0          # near-expiry filter for the headline log metric
CLAMP_LOG = float(np.log(1e-8))    # = -18.420...; matches the training clamp floor
SQERR_TAIL_THRESHOLD = 10.0        # "tail" = per-sample squared log-error > this


# ==========================================================================
# SPECTRAL CONVOLUTION + FNO ENCODER  (identical across all 12 train scripts)
# ==========================================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO_MarketEncoder(nn.Module):
    """2D FNO over the market-state grid -> latent/basis vector.

    Bodies match across both readouts, but the FINAL activation differs and must
    be set per readout: model A applies `F.silu(fc2(x))` (train_*.py:133), model B
    returns the raw `fc2(x)` (train_*_don.py:133) because the DeepONet dot product
    needs a signed branch basis. Pass `final_silu=False` for model B. The final
    projection width is named `latent_dim` (MLP-head) vs `p_dim` (DeepONet).
    State-dict key prefix is `encoder.` in model A and `branch.` in model B.
    """
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32, out_dim=64,
                 final_silu=True):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.width = width
        self.final_silu = final_silu
        self.fc0 = nn.Conv2d(1, width, 1)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        flat_dim = width * grid_h * grid_w
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.grid_h, self.grid_w)
        x = self.fc0(x)
        x = F.silu(self.conv0(x) + self.w0(x))
        x = F.silu(self.conv1(x) + self.w1(x))
        x = F.silu(self.conv2(x) + self.w2(x))
        x = F.silu(self.conv3(x) + self.w3(x))
        x = x.view(x.shape[0], -1)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return F.silu(x) if self.final_silu else x


# ==========================================================================
# READOUTS:  A = MLP-head (softplus),  B = DeepONet trunk (+ branch.trunk dot)
# ==========================================================================

class MLPVolHead(nn.Module):
    """Model A head. State-dict prefix `head.net.*`. Softplus inside forward."""
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent, log_moneyness, T_years):
        x = torch.cat([latent, log_moneyness, T_years], dim=-1)
        return F.softplus(self.net(x))


class DeepONetTrunk(nn.Module):
    """Model B trunk. State-dict prefix `trunk.net.*`. No positivity here."""
    def __init__(self, p_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, p_dim),
        )

    def forward(self, log_moneyness, T_years):
        return self.net(torch.cat([log_moneyness, T_years], dim=-1))


# ==========================================================================
# ANALYTICAL BLACK-SCHOLES PRICER  (verbatim from the train scripts)
# ==========================================================================

def _norm_cdf(x):
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    return 0.5 * (1.0 + torch.erf(x / sqrt2))


def bs_normalized_price(log_moneyness, T_years, r, q, sigma, option_type="call"):
    M = torch.exp(log_moneyness)
    sqrt_T = torch.sqrt(torch.clamp(T_years, min=1e-10))
    sigma_safe = torch.clamp(sigma, min=1e-10)
    d1 = (log_moneyness + (r - q + 0.5 * sigma_safe ** 2) * T_years) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    if option_type == "call":
        price = M * torch.exp(-q * T_years) * _norm_cdf(d1) - torch.exp(-r * T_years) * _norm_cdf(d2)
    else:
        price = torch.exp(-r * T_years) * _norm_cdf(-d2) - M * torch.exp(-q * T_years) * _norm_cdf(-d1)
    return price


def bs_log_normalized_price(log_moneyness, T_years, r, q, sigma, option_type="call"):
    """Numerically-stable log(V/K) via log_ndtr (eval-only; gradient explodes ATM)."""
    sqrt_T = torch.sqrt(torch.clamp(T_years, min=1e-10))
    sigma_safe = torch.clamp(sigma, min=1e-10)
    d1 = (log_moneyness + (r - q + 0.5 * sigma_safe ** 2) * T_years) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    if option_type == "call":
        log_a = log_moneyness - q * T_years + torch.special.log_ndtr(d1)
        log_b = -r * T_years + torch.special.log_ndtr(d2)
    else:
        log_a = -r * T_years + torch.special.log_ndtr(-d2)
        log_b = log_moneyness - q * T_years + torch.special.log_ndtr(-d1)
    diff = torch.clamp(log_b - log_a, max=-1e-10)
    return log_a + torch.log(-torch.expm1(diff))


# ==========================================================================
# COMBINED MODELS  (attribute names chosen to match the saved state dicts)
# ==========================================================================

class VolEstimatorModelA(nn.Module):
    """MLP-head readout (`train_*.py`). state_dict: encoder.*, head.*"""
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32,
                 latent_dim=128, head_hidden=128, option_type="call",
                 use_stable_log=False):
        super().__init__()
        self.encoder = FNO_MarketEncoder(grid_h, grid_w, modes1, modes2, width, out_dim=latent_dim)
        self.head = MLPVolHead(latent_dim=latent_dim, hidden=head_hidden)
        self.use_stable_log = use_stable_log
        self.option_type = option_type

    def sigma(self, branch, log_moneyness, T_years):
        return self.head(self.encoder(branch), log_moneyness, T_years)

    def forward(self, branch, log_moneyness, T_years, r, q):
        sigma_hat = self.sigma(branch, log_moneyness, T_years)
        return sigma_hat, _price_to_logv(self, sigma_hat, log_moneyness, T_years, r, q)


class VolDeepONetModelB(nn.Module):
    """DeepONet readout (`train_*_don.py`). state_dict: branch.*, trunk.*, sigma_bias"""
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32,
                 p_dim=128, trunk_hidden=128, sigma_floor=1e-6, option_type="call",
                 use_stable_log=False):
        super().__init__()
        self.branch = FNO_MarketEncoder(grid_h, grid_w, modes1, modes2, width, out_dim=p_dim,
                                        final_silu=False)
        self.trunk = DeepONetTrunk(p_dim=p_dim, hidden=trunk_hidden)
        self.sigma_bias = nn.Parameter(torch.tensor([-1.50]))
        self.sigma_floor = sigma_floor
        self.use_stable_log = use_stable_log
        self.option_type = option_type

    def sigma(self, branch, log_moneyness, T_years):
        b = self.branch(branch)
        t = self.trunk(log_moneyness, T_years)
        raw = torch.sum(b * t, dim=1, keepdim=True) + self.sigma_bias
        return F.softplus(raw) + self.sigma_floor

    def forward(self, branch, log_moneyness, T_years, r, q):
        sigma_hat = self.sigma(branch, log_moneyness, T_years)
        return sigma_hat, _price_to_logv(self, sigma_hat, log_moneyness, T_years, r, q)


def _price_to_logv(model, sigma_hat, log_moneyness, T_years, r, q):
    if model.use_stable_log:
        return bs_log_normalized_price(log_moneyness, T_years, r, q, sigma_hat, model.option_type)
    price = bs_normalized_price(log_moneyness, T_years, r, q, sigma_hat, model.option_type)
    return torch.log(torch.clamp(price, min=1e-8))


# ==========================================================================
# CONFIG -> MODEL  (arch inferred from results dir name / config keys)
# ==========================================================================

def infer_arch(results_dir: Path, config: dict) -> str:
    """Return 'B' for the DeepONet readout, else 'A'."""
    if str(results_dir).rstrip("/\\").endswith("_don"):
        return "B"
    return "B" if "p_dim" in config else "A"


def build_model(config: dict, arch: str, use_stable_log: bool = False) -> nn.Module:
    common = dict(
        grid_h=config["grid_h"], grid_w=config["grid_w"],
        modes1=config["modes1"], modes2=config["modes2"],
        width=config.get("fno_width", 32), option_type=config["option_type"],
        use_stable_log=use_stable_log,
    )
    if arch == "B":
        return VolDeepONetModelB(
            p_dim=config.get("p_dim", 128),
            trunk_hidden=config.get("trunk_hidden", 128),
            sigma_floor=config.get("sigma_floor", 1e-6),
            **common,
        )
    return VolEstimatorModelA(
        latent_dim=config.get("latent_dim", 128),
        head_hidden=config.get("head_hidden", 128),
        **common,
    )


def load_run(results_dir, use_stable_log=False, weights="best_model.pth", device="cpu"):
    """Load (config, model) for a `results_*/` dir; weights loaded strict=True."""
    results_dir = Path(results_dir)
    config = json.loads((results_dir / "config.json").read_text())
    arch = infer_arch(results_dir, config)
    model = build_model(config, arch, use_stable_log=use_stable_log).to(device)
    state = torch.load(str(results_dir / weights), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return config, arch, model


# ==========================================================================
# HDF5 TEST-SPLIT LOADER  (reads the extra columns the train loader skips)
# ==========================================================================

def load_test_split(h5_path, branch_key, split="test", max_contracts=None, seed=42):
    """Return a dict of numpy arrays for one split.

    Adds `moneyness`, `normalized_price`, `date` (the columns the train-time
    `H5BranchDataset` ignores) so baselines / figures can use them. `branch_u`
    (the gridded IV surface) is always returned for the interpolation baseline.
    """
    split_val = {"train": 0, "val": 1, "test": 2}[split]
    with h5py.File(h5_path, "r") as f:
        idx = np.where(f["split_id"][:] == split_val)[0]
        if max_contracts is not None and len(idx) > max_contracts:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_contracts, replace=False))
        trunk = f["trunk_y"][idx]
        out = {
            "branch": f[branch_key][idx],
            "branch_u": f["branch_u"][idx],
            "log_m": trunk[:, 0:1],
            "T": trunk[:, 1:2],
            "r": trunk[:, 2:3],
            "q": trunk[:, 3:4],
            "target_v_log": f["target_v_log"][idx],
            "moneyness": f["moneyness"][idx],
            "normalized_price": f["normalized_price"][idx],
            "date": f["date"][idx],
            "indices": idx,
        }
        # Quote-liquidity columns exist in v4/v5 only -- presence-checked so v3 runs.
        for opt in ("best_bid", "best_offer", "spread_norm", "half_spread_norm"):
            if opt in f:
                out[opt] = f[opt][idx].astype(np.float64).ravel()
    return out


# ==========================================================================
# QUOTE-LIQUIDITY REPORTING STRATA  (NOT a filter -- nothing is ever dropped)
# ==========================================================================
#
# These masks exist so the manuscript can show the headline numbers are not
# carried by retained zero-bid or wide-spread quotes. They SUBSET the metric
# report only: no caller removes rows from training, from a split, or from the
# "all rows" metrics. Turning any of these into an actual sample filter is a
# separate, explicit dataset decision that has NOT been made -- if you find
# yourself writing `rows = rows[mask]` with one of these, stop.
#
# relative spread = (ask - bid) / mid, mid = (bid + ask)/2. With bid >= 0 this is
# bounded above by 2, so a threshold > 2 selects everything and says nothing.
LIQUIDITY_STRATA_DOC = {
    "bid_pos": "best_bid > 0 (excludes zero-bid quotes from the REPORT only)",
    "relspread_le1": "(ask - bid) / mid <= 1 (bounded by 2 when bid >= 0)",
    "bid_pos_and_relspread_le1": "both of the above",
    "_not_a_filter": "Reporting strata only. No row is dropped anywhere on liquidity.",
}


def liquidity_masks(best_bid, best_offer):
    """-> {stratum: bool mask} for the quote-liquidity REPORTING strata.

    Returns {} when either column is unavailable (v3 samples), so every caller
    degrades to "no liquidity strata reported" instead of failing.
    """
    if best_bid is None or best_offer is None:
        return {}
    bid = np.asarray(best_bid, dtype=np.float64).ravel()
    ask = np.asarray(best_offer, dtype=np.float64).ravel()
    mid = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mid > 0, (ask - bid) / mid, np.inf)
    bid_pos = bid > 0
    narrow = rel <= 1.0
    return {"bid_pos": bid_pos, "relspread_le1": narrow,
            "bid_pos_and_relspread_le1": bid_pos & narrow}


# ==========================================================================
# DATASET SELECTION  (manifest-driven -- never mix samples silently)
# ==========================================================================
#
# TWO INDEPENDENT AXES, never inferred from one another:
#   schema_version  = tensor structure of the HDF5 (v3 -> v4; v5 files are still
#                     schema "v4", the layout did not change).
#   dataset_version = which SAMPLE the rows are (v3 calendar-T; v4 settlement-T,
#                     T > 1 day, unfiltered quotes; v5 = v4 plus the zero-tolerance
#                     static-no-arbitrage midpoint quote filter).
# `wrds_data_2020-2025/DATASET_MANIFEST.json` is the single source of truth for
# which dataset versions exist, where their files are, what they hash to, and
# which one is canonical. Nothing here hard-codes a hash or a canonical version.

# $DL_DATASET_MANIFEST overrides the path (smoke tests / a relocated data dir).
MANIFEST_PATH = Path(os.environ.get(
    "DL_DATASET_MANIFEST", PROJECT_ROOT / "wrds_data_2020-2025" / "DATASET_MANIFEST.json"))

# Smoke builds are deliberately NOT in the manifest (they are throwaway subsamples,
# not samples anyone may publish). They keep a tiny local table: (subdir, template,
# dataset_version, schema_version) -- both versions stated, neither inferred.
SMOKE_VERSIONS = {
    "v4smoke": ("wrds_data_2020-2025/smoke_v4", "deeponet_tensors_%s_v4_smoke.h5", "v4", "v4"),
    "v5smoke": ("wrds_data_2020-2025/smoke_v5", "deeponet_tensors_%s_v5_smoke.h5", "v5", "v4"),
}
# Selectable on the command line. Presence here is NOT a claim that the version
# exists -- resolution goes through the manifest and fails loudly if it doesn't.
DATASET_VERSION_CHOICES = ("v3", "v4", "v5", "v4smoke", "v5smoke")

_MANIFEST_CACHE = {}


def load_manifest(path=None) -> dict:
    """Read DATASET_MANIFEST.json (cached). Missing/corrupt manifest = hard error."""
    p = Path(path or MANIFEST_PATH)
    key = str(p)
    if key not in _MANIFEST_CACHE:
        if not p.exists():
            raise SystemExit(
                "MISSING DATASET MANIFEST: %s\n"
                "  Every analysis resolves dataset identity, paths, hashes and which\n"
                "  version is canonical from this file. There is deliberately no\n"
                "  hard-coded fallback -- a stale built-in default is how a superseded\n"
                "  sample reaches a manuscript table. Build/restore the manifest first."
                % p)
        try:
            _MANIFEST_CACHE[key] = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit("CORRUPT DATASET MANIFEST %s: %s" % (p, exc))
    return _MANIFEST_CACHE[key]


def manifest_entry(version: str, path=None) -> dict:
    """The manifest's record for one dataset version; absent version = hard error."""
    man = load_manifest(path)
    datasets = man.get("datasets", {})
    if version not in datasets:
        raise SystemExit(
            "UNKNOWN dataset-version %r: not in %s (known: %s).\n"
            "  Refusing to guess a path or a hash for it."
            % (version, Path(path or MANIFEST_PATH), ", ".join(sorted(datasets)) or "(none)"))
    return datasets[version]


def canonical_version(path=None) -> str:
    """Which dataset version the manifest declares canonical for training/T1."""
    man = load_manifest(path)
    if not man.get("canonical"):
        raise SystemExit("DATASET MANIFEST has no `canonical` key: %s"
                         % Path(path or MANIFEST_PATH))
    return str(man["canonical"])


def dataset_versions(version: str, manifest_path=None):
    """-> (dataset_version, schema_version) for a selector. Read, never inferred."""
    if version in SMOKE_VERSIONS:
        return SMOKE_VERSIONS[version][2], SMOKE_VERSIONS[version][3]
    e = manifest_entry(version, manifest_path)
    return str(e["dataset_version"]), str(e["schema_version"])


def dataset_h5(version: str, option_type: str, data_dir=None, manifest_path=None) -> Path:
    """Resolve the HDF5 path for a (dataset version, option type) pair."""
    if version in SMOKE_VERSIONS:
        sub, tmpl = SMOKE_VERSIONS[version][:2]
        root = Path(data_dir) if data_dir else PROJECT_ROOT / sub
        return root / (tmpl % option_type)
    files = manifest_entry(version, manifest_path).get("files", {})
    if option_type not in files:
        raise SystemExit("DATASET MANIFEST: version %s has no %r file entry"
                         % (version, option_type))
    p = Path(files[option_type]["path"])
    if data_dir:                       # --h5-dir relocates the same file
        return Path(data_dir) / p.name
    # Manifest paths are relative to the MANIFEST'S OWN directory, not the repo
    # root: the manifest lives in wrds_data_2020-2025/ alongside the files it
    # describes, and records them by bare filename. Resolving against
    # PROJECT_ROOT instead sent every lookup to <repo>/deeponet_tensors_*.h5 and
    # failed with FileNotFoundError. Manifest-relative also keeps the data
    # directory relocatable -- move the pair together and paths still resolve.
    base = (Path(manifest_path) if manifest_path else MANIFEST_PATH).parent
    return p if p.is_absolute() else base / p


def h5_attr(h5_path, name, default=None):
    """One root attr as str (or `default` when absent / file missing)."""
    p = Path(h5_path)
    if not p.exists():
        return default
    with h5py.File(p, "r") as f:
        v = f.attrs.get(name, None)
    if v is None:
        return default
    return v.decode() if isinstance(v, bytes) else str(v)


def dataset_provenance(version: str, option_type: str, data_dir=None,
                       manifest_path=None) -> dict:
    """Path + BOTH version axes + canonical status, for stamping into every output.

    Cross-checks the on-file attrs against the manifest and refuses any
    disagreement. v3 files predate both attrs, so an absent `schema_version` is
    reported as "v3 (attr absent)" rather than guessed silently; an absent
    `dataset_version` attr (every v4 file predates that attr too) is taken from
    the manifest, never from the schema.
    """
    p = dataset_h5(version, option_type, data_dir, manifest_path)
    want_ds, want_schema = dataset_versions(version, manifest_path)
    schema = h5_attr(p, "schema_version")
    on_file_ds = h5_attr(p, "dataset_version")

    if schema is None:
        schema = "v3 (attr absent)"
    elif schema != want_schema:
        raise SystemExit(
            "dataset-version %s expects schema_version %r but %s carries %r -- "
            "refusing to mix samples" % (version, want_schema, p, schema))
    if on_file_ds is not None and on_file_ds != want_ds:
        raise SystemExit(
            "dataset-version %s expects dataset_version %r but %s carries %r -- "
            "refusing to mix samples" % (version, want_ds, p, on_file_ds))

    prov = {"dataset_version": want_ds, "h5_path": str(p), "schema_version": str(schema),
            "selector": version}
    if version not in SMOKE_VERSIONS:
        e = manifest_entry(version, manifest_path)
        prov["expected_sha256"] = e.get("files", {}).get(option_type, {}).get("sha256")
        prov["quote_filter"] = e.get("quote_filter")
        prov["quote_filter_tolerance"] = e.get("quote_filter_tolerance")
        prov["dataset_status"] = e.get("status")
        prov["is_canonical_version"] = (want_ds == canonical_version(manifest_path))
    else:
        prov["is_canonical_version"] = False
        prov["dataset_status"] = "smoke subsample; never publishable"
    return prov


def output_tag(version: str) -> str:
    """Filename suffix so a v5 run cannot overwrite the v3/v4-sample outputs."""
    return "" if version == "v3" else "_" + version


def add_dataset_arg(ap):
    ap.add_argument("--dataset-version", required=True, choices=DATASET_VERSION_CHOICES,
                    help="which sample to read (REQUIRED -- no default, because a silent "
                         "default is how stale numbers reach a manuscript). Paths, hashes "
                         "and canonical status come from wrds_data_2020-2025/"
                         "DATASET_MANIFEST.json; v4smoke/v5smoke are the small smoke builds.")


def resolve_h5_path(config: dict, manifest_path=None) -> Path:
    """The HDF5 a trained run was fed, as reachable from THIS machine.

    config.json stores an absolute path from the training machine. Fall backs, in
    order: same basename in the data dir, then the manifest file whose sha256
    equals the config's. There is deliberately no blind fall back to the v3 file
    (that silently swapped the sample under the eval).
    """
    p = Path(config["h5_path"])
    if p.exists():
        return p
    local = PROJECT_ROOT / "wrds_data_2020-2025" / p.name
    if local.exists():
        return local
    sha = config.get("h5_sha256")
    if sha:
        for ver, e in load_manifest(manifest_path).get("datasets", {}).items():
            for opt, rec in e.get("files", {}).items():
                if rec.get("sha256") == sha:
                    cand = Path(rec["path"])
                    cand = cand if cand.is_absolute() else PROJECT_ROOT / cand
                    if cand.exists():
                        return cand
    raise SystemExit(
        "CANNOT LOCATE the HDF5 this run was trained on:\n"
        "  config.json h5_path = %s (does not exist here)\n"
        "  no %s in wrds_data_2020-2025/, and no manifest file matches sha256 %s.\n"
        "  Refusing to substitute a different sample."
        % (p, p.name, str(sha)[:12] + "..." if sha else "(none recorded)"))


# ==========================================================================
# CANONICAL METRIC SCHEMA  (factored from the train-script eval block so model
# eval, baselines and ablations all emit the SAME json keys)
# ==========================================================================

def compute_metrics(log_pred, log_target, T, log_m=None, n_seconds=None,
                    best_bid=None, best_offer=None):
    """All inputs 1-D numpy. Mirrors train_vol_surface.py:613-691 exactly.

    `best_bid`/`best_offer` (v4/v5 only, optional) add the quote-liquidity
    REPORTING strata under metrics["liquidity_strata"]. They never change the
    headline numbers and never drop a row -- see LIQUIDITY_STRATA_DOC.
    """
    log_pred = np.asarray(log_pred, dtype=np.float64).ravel()
    log_target = np.asarray(log_target, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    n = log_pred.size

    resid = log_pred - log_target
    mse_log = float(np.mean(resid ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_target - log_target.mean()) ** 2))
    r2_log = 1.0 - ss_res / (ss_tot + 1e-10)

    p_pred = np.exp(log_pred)
    p_true = np.exp(log_target)
    mse_price = float(np.mean((p_pred - p_true) ** 2))
    mae_price = float(np.mean(np.abs(p_pred - p_true)))
    ss_res_p = float(np.sum((p_true - p_pred) ** 2))
    ss_tot_p = float(np.sum((p_true - p_true.mean()) ** 2))
    r2_price = 1.0 - ss_res_p / (ss_tot_p + 1e-10)

    keep = T > T_MIN_YEARS
    n_keep = int(keep.sum())
    if n_keep > 0:
        tk, pk = log_target[keep], log_pred[keep]
        ss_res_k = float(np.sum((tk - pk) ** 2))
        ss_tot_k = float(np.sum((tk - tk.mean()) ** 2))
        r2_log_filtered = 1.0 - ss_res_k / (ss_tot_k + 1e-10)
        mse_log_filtered = ss_res_k / n_keep
    else:
        r2_log_filtered = float("nan")
        mse_log_filtered = float("nan")

    sq = resid ** 2
    tail = sq > SQERR_TAIL_THRESHOLD
    n_tail = int(tail.sum())
    metrics = {
        "n_test": int(n),
        "r2_price": r2_price,
        "r2_log_filtered": r2_log_filtered,   # headline
        "r2_log_full": r2_log,
        "rmse_log": float(np.sqrt(mse_log)),
        "rmse_price": float(np.sqrt(mse_price)),
        "mae_price": mae_price,
        "mse_log": mse_log,
        "mse_log_filtered": mse_log_filtered,
        "n_keep_filtered": n_keep,
        "n_near_expiry_dropped": int(n - n_keep),
        "n_tail_sqerr_gt10": n_tail,
        "tail_sse_share": float(sq[tail].sum() / sq.sum()) if n_tail else 0.0,
        "clamp_hit_count": int((log_pred < -18.42).sum()),  # match train eval's hard -18.42
    }
    if n_seconds is not None and n_seconds > 0:
        metrics["throughput_contracts_per_s"] = float(n / n_seconds)

    strata = liquidity_masks(best_bid, best_offer)
    if strata:
        # Sub-REPORT only: recompute the same schema on the subset. The metrics
        # above are computed on ALL rows and stay untouched -- no row is dropped.
        keep_keys = ("n_test", "r2_price", "r2_log_filtered", "r2_log_full",
                     "rmse_log", "rmse_price", "mae_price")
        sub = {}
        for name, m in strata.items():
            if not m.any():
                sub[name] = {"n_test": 0}
                continue
            s = compute_metrics(log_pred[m], log_target[m], T[m])
            sub[name] = {k: s[k] for k in keep_keys}
        metrics["liquidity_strata"] = sub
        metrics["liquidity_strata_note"] = LIQUIDITY_STRATA_DOC
    return metrics


@torch.no_grad()
def model_predict_logv(model, data, device="cpu", batch_size=4096):
    """Batched forward over a load_test_split() dict -> (log_pred, sigma_hat) arrays."""
    model.eval()
    branch = torch.as_tensor(data["branch"], dtype=torch.float32)
    log_m = torch.as_tensor(data["log_m"], dtype=torch.float32)
    T = torch.as_tensor(data["T"], dtype=torch.float32)
    r = torch.as_tensor(data["r"], dtype=torch.float32)
    q = torch.as_tensor(data["q"], dtype=torch.float32)
    preds, sigmas = [], []
    for i in range(0, branch.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        s, lv = model(branch[sl].to(device), log_m[sl].to(device), T[sl].to(device),
                      r[sl].to(device), q[sl].to(device))
        preds.append(lv.cpu().numpy().ravel())
        sigmas.append(s.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(sigmas)
