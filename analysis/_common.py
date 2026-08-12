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
    return out


def resolve_h5_path(config: dict) -> Path:
    """config.json stores an absolute h5_path from the training machine; fall back
    to the repo-relative path if it doesn't exist here."""
    p = Path(config["h5_path"])
    if p.exists():
        return p
    name = "deeponet_tensors_%s.h5" % config["option_type"]
    return PROJECT_ROOT / "wrds_data_2020-2025" / name


# ==========================================================================
# CANONICAL METRIC SCHEMA  (factored from the train-script eval block so model
# eval, baselines and ablations all emit the SAME json keys)
# ==========================================================================

def compute_metrics(log_pred, log_target, T, log_m=None, n_seconds=None):
    """All inputs 1-D numpy. Mirrors train_vol_surface.py:613-691 exactly."""
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
