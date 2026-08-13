"""
Train FNO model on put options using SPX OHLCV history (spot_history).
V3.1: Per-query implicit IV surface (was: single scalar sigma per market state).

Architecture:
  Encoder: 2D FNO on spot history grid (21x5) -> market-state latent vector
  Head:    MLP([latent, log_moneyness, T_years]) -> sigma_hat (positive scalar)
  Pricing: bs_normalized_price([log_moneyness, T_years, r, q, sigma_hat]) -> V/K

Loss: Data MSE on log(V/K). No PDE / arbitrage / BS-anchor terms.

Why the change: a single scalar sigma cannot price both ATM and deep-OTM
contracts under the same market state (volatility smile). The conditional head
lets the model emit a different sigma for each (log_m, T) query against the
same market-state latent, recovering the smile/skew that constant-vol BS cannot
capture.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ==========================================
# SPECTRAL CONVOLUTION
# ==========================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ==========================================
# NETWORK 1: FNO MARKET-STATE ENCODER (SiLU)
# ==========================================

class FNO_MarketEncoder(nn.Module):
    """FNO-based encoder: 2D market-state grid -> latent vector.

    Takes a 2D grid, passes through 4 FNO blocks with residual connections,
    then projects to a fixed-size latent. The latent is consumed downstream
    by ConditionalVolHead together with the per-contract (log_m, T) query.
    Uses SiLU activation throughout.
    """
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32, latent_dim=64):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.width = width
        self.latent_dim = latent_dim

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
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.grid_h, self.grid_w)
        x = self.fc0(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.silu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.silu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.silu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.silu(x1 + x2)

        x = x.view(x.shape[0], -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return x  # (batch, latent_dim)


# ==========================================
# CONDITIONAL VOL HEAD
# ==========================================

class ConditionalVolHead(nn.Module):
    """MLP mapping (market_state_latent, log_moneyness, T_years) -> sigma_hat.

    Lets the model emit a different sigma per contract query against the same
    market-state latent, so it can fit the volatility smile/skew that a single
    scalar sigma cannot represent. Output is softplus-positive.
    """
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


# ==========================================
# ANALYTICAL BLACK-SCHOLES PRICER
# ==========================================

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
    """Numerically stable log(V/K) via log_ndtr, avoiding the fp32 catastrophic
    cancellation in M*N(d1) - exp(-rT)*N(d2) for short-T / near-ATM contracts.

    price/K = exp(log_a) - exp(log_b) with log_a >= log_b. Computed as
    log_a + log(-expm1(diff)), diff = log_b - log_a <= 0. `expm1` keeps full
    precision for tiny |diff| (where the naive `M*N - e^{-rT}*N` collapses).

    NOTE: the gradient `1/(-expm1(diff))` explodes as diff -> 0 (near-ATM).
    This is safe for evaluation (no grads). For training it needs a softer
    `diff` clamp and/or per-sample grad clipping.
    """
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


# ==========================================
# COMBINED MODEL
# ==========================================

class VolEstimatorModel(nn.Module):
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32,
                 latent_dim=64, head_hidden=64, option_type="call",
                 use_stable_log=False):
        super().__init__()
        self.encoder = FNO_MarketEncoder(
            grid_h, grid_w, modes1, modes2, width, latent_dim=latent_dim,
        )
        self.head = ConditionalVolHead(latent_dim=latent_dim, hidden=head_hidden)
        # When True, price log(V/K) with the stable log_ndtr path instead of
        # log(clamp(price, 1e-8)). Intended for eval; see bs_log_normalized_price.
        self.use_stable_log = use_stable_log
        self.option_type = option_type

    def forward(self, branch, log_moneyness, T_years, r, q):
        latent = self.encoder(branch)
        sigma_hat = self.head(latent, log_moneyness, T_years)
        if self.use_stable_log:
            log_v_hat = bs_log_normalized_price(
                log_moneyness, T_years, r, q, sigma_hat, self.option_type,
            )
        else:
            price = bs_normalized_price(log_moneyness, T_years, r, q, sigma_hat, self.option_type)
            log_v_hat = torch.log(torch.clamp(price, min=1e-8))
        return sigma_hat, log_v_hat


# ==========================================
# TRAINING LOSS (data-only)
# ==========================================

def compute_loss(model, branch, log_moneyness, T_years, r, q, target_v_log):
    sigma_hat, log_v_hat = model(branch, log_moneyness, T_years, r, q)
    # Huber (smooth-L1) loss: quadratic for |residual| < delta, linear beyond.
    # This caps the gradient contribution of the ~1.5% near-expiry / near-ATM
    # tail (where log-price is ill-conditioned and residuals are huge), so the
    # bulk of contracts is not drowned out. delta=1.0 keeps the bulk (residuals
    # << 1) in the usual squared-error regime.
    # loss_data = F.huber_loss(log_v_hat, target_v_log, delta=1.0)

    # --- Old MSE loss (switch back by uncommenting this and removing Huber): ---
    loss_data = F.mse_loss(log_v_hat, target_v_log)
    return loss_data, {"data": loss_data.item(), "total": loss_data.item()}


# ==========================================
# DATASET
# ==========================================

class H5BranchDataset(Dataset):
    """Loads one branch dataset from HDF5, filtered by split_id."""
    def __init__(self, h5_path, split, branch_key):
        split_val = {"train": 0, "val": 1, "test": 2}[split]

        with h5py.File(h5_path, "r") as f:
            split_ids = f["split_id"][:]
            mask = split_ids == split_val
            self.indices = np.where(mask)[0]
            self.length = len(self.indices)

            self.branch = f[branch_key][self.indices]
            self.trunk_y = f["trunk_y"][self.indices]
            self.target = f["target_v_log"][self.indices]

        print(f"Loaded {split}: {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.branch[idx], dtype=torch.float32),
            torch.tensor(self.trunk_y[idx], dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.float32),
        )


def make_loader(dataset, batch_size, shuffle):
    def collate_fn(batch):
        branch = torch.stack([b[0] for b in batch])
        trunk_y = torch.stack([b[1] for b in batch])
        target = torch.stack([b[2] for b in batch])
        return (
            branch,
            trunk_y[:, 0:1],   # log_moneyness
            trunk_y[:, 1:2],   # T_years
            trunk_y[:, 2:3],   # r
            trunk_y[:, 3:4],   # q
            target,             # target_v_log
        )
    return DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      drop_last=shuffle)


# ==========================================
# EVALUATION
# ==========================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    for branch, log_m, T, r, q, target in loader:
        branch, log_m, T, r, q, target = [b.to(device) for b in (branch, log_m, T, r, q, target)]
        _, log_v_hat = model(branch, log_m, T, r, q)
        total_mse += F.mse_loss(log_v_hat, target, reduction="sum").item()
        total_samples += target.size(0)
    model.train()
    return total_mse / total_samples


# ==========================================
# TRAINING
# ==========================================

def train_model(config):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs_adam = config["epochs_adam"]
    epochs_finetune = config["epochs_finetune"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    finetune_lr = config["finetune_lr"]
    grad_clip = config["grad_clip"]
    patience_adam = config["patience_adam"]
    patience_finetune = config["patience_finetune"]
    min_finetune_epochs = config["min_finetune_epochs"]
    max_train_batches = config.get("max_train_batches", None)
    eval_only = config.get("eval_only", False)
    fno_width = config["fno_width"]

    h5_path = config["h5_path"]
    branch_key = config["branch_key"]
    grid_h = config["grid_h"]
    grid_w = config["grid_w"]
    modes1 = config["modes1"]
    modes2 = config["modes2"]
    option_type = config["option_type"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config (skip in eval-only so we don't clobber the trained run's config)
    if not eval_only:
        with open(results_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    print("Loading datasets...")
    train_ds = H5BranchDataset(h5_path, "train", branch_key)
    val_ds = H5BranchDataset(h5_path, "val", branch_key)
    test_ds = H5BranchDataset(h5_path, "test", branch_key)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size * 2, shuffle=False)
    test_loader = make_loader(test_ds, batch_size * 2, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    latent_dim = config.get("latent_dim", 64)
    head_hidden = config.get("head_hidden", 64)
    use_stable_log = config.get("use_stable_log", False)
    model = VolEstimatorModel(
        grid_h, grid_w, modes1, modes2, fno_width,
        latent_dim=latent_dim, head_hidden=head_hidden,
        option_type=option_type, use_stable_log=use_stable_log,
    ).to(device)
    if use_stable_log:
        print("  [stable-log] pricing log(V/K) via log_ndtr (no 1e-8 clamp floor)")
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    total_params = enc_params + head_params
    print(
        f"VolEstimatorModel initialized: fno_width={fno_width}, "
        f"latent_dim={latent_dim}, head_hidden={head_hidden}"
    )
    print(f"  Encoder params: {enc_params:,}  Head params: {head_params:,}")
    print(f"  Total model parameters: {total_params:,}")

    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer_adam, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    all_train_losses = []
    all_val_losses = []

    # ============ PHASE 1: ADAM ============
    print("\n" + "=" * 60)
    print("PHASE 1: Adam Optimizer")
    print("=" * 60)

    epochs_no_improve = 0
    phase1_epochs = 0

    for epoch in range(0 if eval_only else epochs_adam):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        comps = {"data": 0.0}
        n_batches_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            n_batches_run += 1

            branch, log_m, T, r, q, target = [b.to(device) for b in batch]

            optimizer_adam.zero_grad()
            loss, loss_dict = compute_loss(model, branch, log_m, T, r, q, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer_adam.step()

            total_loss += loss_dict["total"]
            for k in comps:
                comps[k] += loss_dict[k]

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Batch Loss: {loss_dict['total']:.6f}"
                )

        for k in comps:
            comps[k] /= n_batches_run

        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        current_lr = optimizer_adam.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches_run

        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss)
        phase1_epochs += 1

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(results_dir / "best_model_phase1.pth"))
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"\nAdam Epoch {epoch+1}/{epochs_adam} | Time: {epoch_time:.1f}s | Base LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_loss:.6f} | Val MSE: {val_loss:.6f}{improved}")
        print(f"  Components - Data: {comps['data']:.4f}")
        print(f"  Epochs without improvement: {epochs_no_improve}/{patience_adam}")
        print("-" * 60)

        if epochs_no_improve >= patience_adam:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

        torch.cuda.empty_cache()

    print(f"\n*** Adam Phase Complete. Best Val MSE: {best_val:.6f} ***")

    if not eval_only:
        print("Loading best model from Adam phase for fine-tuning...")
        if (results_dir / "best_model_phase1.pth").exists():
            model.load_state_dict(torch.load(str(results_dir / "best_model_phase1.pth")))
        torch.save(model.state_dict(), str(results_dir / "best_model.pth"))

    # ============ PHASE 2: FINE-TUNE ============
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning with Adam (Low LR)")
    print("=" * 60)

    optimizer_ft = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-6)
    epochs_no_improve = 0

    for epoch in range(0 if eval_only else epochs_finetune):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        comps = {"data": 0.0}
        n_batches_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            n_batches_run += 1

            branch, log_m, T, r, q, target = [b.to(device) for b in batch]

            optimizer_ft.zero_grad()
            loss, loss_dict = compute_loss(model, branch, log_m, T, r, q, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer_ft.step()

            total_loss += loss_dict["total"]
            for k in comps:
                comps[k] += loss_dict[k]

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Batch Loss: {loss_dict['total']:.6f}"
                )

        for k in comps:
            comps[k] /= n_batches_run

        val_loss = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches_run

        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(results_dir / "best_model.pth"))
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"\nFine-tune Epoch {epoch+1}/{epochs_finetune} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_loss:.6f} | Val MSE: {val_loss:.6f}{improved}")
        print("-" * 60)

        torch.cuda.empty_cache()

        if epoch >= min_finetune_epochs - 1 and epochs_no_improve >= patience_finetune:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best Validation MSE: {best_val:.6f}")
    print("=" * 60)

    if not eval_only:
        torch.save(model.state_dict(), str(results_dir / "final_model.pth"))

    # ============ TEST EVALUATION ============
    print("\n" + "=" * 60)
    print("RUNNING FINAL TEST SET EVALUATION")
    print("=" * 60)

    if (results_dir / "best_model.pth").exists():
        model.load_state_dict(torch.load(str(results_dir / "best_model.pth")))
    model.eval()

    test_mse_log = 0.0
    test_mae_price = 0.0
    test_mse_price = 0.0
    total_samples = 0
    all_log_targets = []
    all_log_preds = []
    all_log_m = []
    all_T = []

    with torch.no_grad():
        for branch, log_m, T, r, q, target in test_loader:
            branch, log_m, T, r, q, target = [b.to(device) for b in (branch, log_m, T, r, q, target)]
            _, log_v_hat = model(branch, log_m, T, r, q)

            test_mse_log += F.mse_loss(log_v_hat, target, reduction="sum").item()

            v_hat = torch.exp(log_v_hat)
            v_true = torch.exp(target)
            test_mse_price += F.mse_loss(v_hat, v_true, reduction="sum").item()
            test_mae_price += F.l1_loss(v_hat, v_true, reduction="sum").item()

            total_samples += target.size(0)
            all_log_targets.append(target.cpu())
            all_log_preds.append(log_v_hat.cpu())
            all_log_m.append(log_m.cpu())
            all_T.append(T.cpu())

    test_mse_log /= total_samples
    test_rmse_log = np.sqrt(test_mse_log)
    test_mse_price /= total_samples
    test_rmse_price = np.sqrt(test_mse_price)
    test_mae_price /= total_samples

    all_log_targets = torch.cat(all_log_targets).numpy().ravel()
    all_log_preds = torch.cat(all_log_preds).numpy().ravel()
    all_log_m = torch.cat(all_log_m).numpy().ravel()
    all_T = torch.cat(all_T).numpy().ravel()

    ss_res = np.sum((all_log_targets - all_log_preds) ** 2)
    ss_tot = np.sum((all_log_targets - np.mean(all_log_targets)) ** 2)
    r2_log = 1.0 - ss_res / (ss_tot + 1e-10)

    all_price_targets = np.exp(all_log_targets)
    all_price_preds = np.exp(all_log_preds)
    ss_res_p = np.sum((all_price_targets - all_price_preds) ** 2)
    ss_tot_p = np.sum((all_price_targets - np.mean(all_price_targets)) ** 2)
    r2_price = 1.0 - ss_res_p / (ss_tot_p + 1e-10)

    # Near-expiry-filtered R^2(log): log-price is ill-conditioned as T -> 0
    # (a tiny sigma error blows up log-price), so near-expiry contracts dominate
    # the log metric while being economically negligible (see R^2(price)). Report
    # R^2(log) on contracts with more than ~1 trading day to expiry as the
    # "meaningful-domain" number, alongside the full-domain one above.
    T_MIN_YEARS = 1.0 / 365.0
    keep = all_T.ravel() > T_MIN_YEARS
    n_keep = int(keep.sum())
    if n_keep > 0:
        tgt_k = all_log_targets[keep]
        prd_k = all_log_preds[keep]
        ss_res_k = np.sum((tgt_k - prd_k) ** 2)
        ss_tot_k = np.sum((tgt_k - np.mean(tgt_k)) ** 2)
        r2_log_filtered = 1.0 - ss_res_k / (ss_tot_k + 1e-10)
        mse_log_filtered = ss_res_k / n_keep
    else:
        r2_log_filtered = float("nan")
        mse_log_filtered = float("nan")

    log_errs = (all_log_preds - all_log_targets) ** 2
    print(f"log-error percentiles (squared):")
    for p in [50, 90, 99, 99.9, 100]:
        print(f"  p{p}: {np.percentile(log_errs, p):.6f}")
    print(f"#samples with sq.err > 10: {(log_errs > 10.0).sum()} / {len(log_errs)}")
    print(f"clamp-hit count (pred <= log(1.0001e-8)): {(all_log_preds < -18.42).sum()}")

    # ---- Tail diagnostic: characterize the large-error samples ----
    CLAMP_LOG = -18.42  # log(1e-8)
    mask = log_errs > 10.0
    n_tail = int(mask.sum())
    print("\n--- Tail diagnostic (samples with sq.err > 10) ---")
    if n_tail == 0:
        print("  (no samples with sq.err > 10)")
    else:
        tgt, prd = all_log_targets[mask], all_log_preds[mask]
        print(f"  count: {n_tail} / {len(log_errs)} ({100.0 * n_tail / len(log_errs):.2f}%)")
        print(f"  share of total log-SSE from this tail: {log_errs[mask].sum() / log_errs.sum():.3f}")
        print(f"  target_v_log  [min / median / max]: {tgt.min():.3f} / {np.median(tgt):.3f} / {tgt.max():.3f}")
        print(f"  pred_v_log    [min / median / max]: {prd.min():.3f} / {np.median(prd):.3f} / {prd.max():.3f}")
        print(f"  frac of tail with TRUE target < {CLAMP_LOG:.2f} (price < 1e-8): {(tgt < CLAMP_LOG).mean():.3f}")
        print(f"  frac of tail that are clamp-hit preds (pred <= {CLAMP_LOG:.2f}): {(prd < CLAMP_LOG).mean():.3f}")
        print(f"  mean signed log-error (pred - target) on tail: {(prd - tgt).mean():.3f}")
        print(f"  log_moneyness [min / median / max]: {all_log_m[mask].min():.3f} / {np.median(all_log_m[mask]):.3f} / {all_log_m[mask].max():.3f}")
        print(f"  T_years       [min / median / max]: {all_T[mask].min():.4f} / {np.median(all_T[mask]):.4f} / {all_T[mask].max():.4f}")
    # Reference: distribution of ALL targets and how many are below the clamp floor.
    print(f"  [ref] all target_v_log percentiles [p0.1/p1/p50/p99/p99.9]: "
          f"{np.percentile(all_log_targets, 0.1):.2f} / {np.percentile(all_log_targets, 1):.2f} / "
          f"{np.percentile(all_log_targets, 50):.2f} / {np.percentile(all_log_targets, 99):.2f} / "
          f"{np.percentile(all_log_targets, 99.9):.2f}")
    print(f"  [ref] #samples with TRUE target < {CLAMP_LOG:.2f}: "
          f"{(all_log_targets < CLAMP_LOG).sum()} / {len(all_log_targets)} "
          f"({100.0 * (all_log_targets < CLAMP_LOG).mean():.2f}%)")

    print(f"Final Test MSE (log):  {test_mse_log:.6f}")
    print(f"Test MSE (log):        {test_mse_log:.8f}")
    print(f"Test RMSE (log):       {test_rmse_log:.6f}")
    print(f"Test MSE (price):      {test_mse_price:.8f}")
    print(f"Test RMSE (price):     {test_rmse_price:.6f}")
    print(f"Test MAE (price):      {test_mae_price:.6f}")
    print(f"Test R^2 (log):        {r2_log:.6f}")
    print(f"Test R^2 (price):      {r2_price:.6f}")
    print(f"Test R^2 (log, T>1day): {r2_log_filtered:.6f}  "
          f"[{n_keep}/{len(all_T)} kept, {len(all_T) - n_keep} near-expiry dropped]")
    print(f"Test MSE (log, T>1day): {mse_log_filtered:.8f}")

    # ============ SAVE OUTPUTS ============
    if eval_only:
        print("\nEval-only mode: skipping loss-history/plot writes (preserving trained-run artifacts).")
    else:
        print("\nSaving loss history to 'loss_history.txt'...")

        with open(results_dir / "loss_history.txt", "w") as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\n")
            for i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
                f.write(f"{i+1}\t{tl:.6f}\t{vl:.6f}\n")
            f.write("\n--- Test Metrics ---\n")
            f.write(f"Test MSE (log):     {test_mse_log:.8f}\n")
            f.write(f"Test RMSE (log):    {test_rmse_log:.6f}\n")
            f.write(f"Test MSE (price):   {test_mse_price:.8f}\n")
            f.write(f"Test RMSE (price):  {test_rmse_price:.6f}\n")
            f.write(f"Test MAE (price):   {test_mae_price:.6f}\n")
            f.write(f"Test R^2 (log):     {r2_log:.6f}\n")
            f.write(f"Test R^2 (price):   {r2_price:.6f}\n")
            f.write(f"Test R^2 (log, T>1day): {r2_log_filtered:.6f}  ({n_keep}/{len(all_T)} kept)\n")
            f.write(f"Test MSE (log, T>1day): {mse_log_filtered:.8f}\n")

        print("Generating loss curve plot 'loss_plot.png'...")
        plt.figure(figsize=(10, 6))
        plt.plot(all_train_losses, label="Train Loss")
        plt.plot(all_val_losses, label="Val Loss (MSE)")
        if phase1_epochs < len(all_train_losses):
            plt.axvline(x=phase1_epochs - 1, color="r", linestyle="--", label="Start Fine-tuning")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training & Validation Loss (Test RMSE log: {test_rmse_log:.6f})")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(results_dir / "loss_plot.png")
        plt.close()

    print("Execution Finished.")


# ==========================================
# CONFIG & ENTRY POINT
# ==========================================

def get_config():
    """Return the default configuration for this script."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    h5_path = str(project_root / "wrds_data_2020-2025" / "deeponet_tensors_put_v5.h5")

    return {
        "seed": 42,
        "epochs_adam": 100,
        "epochs_finetune": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "finetune_lr": 1e-5,
        "grad_clip": 1.0,
        "patience_adam": 5,
        "patience_finetune": 5,
        "min_finetune_epochs": 10,
        "max_train_batches": None,
        "fno_width": 32,
        "latent_dim": 128,
        "head_hidden": 128,
        "h5_path": h5_path,
        "branch_key": "spot_history",
        "grid_h": 21,
        "grid_w": 5,
        "modes1": 6,
        "modes2": 2,
        "option_type": "put",
        "results_dir": str(script_dir / "results_spot_history_v5"),
    }


def _sha256_of(path):
    """SHA-256 of a file, read in chunks (works on multi-GB HDF5)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_provenance(cfg):
    """Validate the input HDF5 is a v4 build, then record its provenance in cfg.

    This branch does not consume `vix_history`, so there is deliberately NO gate
    on the VIX source series / CSV hash (that would be a false constraint here).
    What this branch *is* sensitive to is the v4 dataset change itself: the
    settlement-aware maturity basis, the T > 1 day export filter (no structurally
    unfittable T=0 rows), and AM/PM contract identity. So the gate is
    `schema_version == "v4"` + `dataset_version == "v5"` (plus the quote-filter
    attrs), and the recorded provenance is the general dataset
    provenance, so a v3-trained and a v4-trained run can never be confused in the
    final table.
    """
    h5 = Path(cfg["h5_path"])
    if not h5.exists():
        raise SystemExit(
            f"REFUSING TO TRAIN: h5_path does not exist: {h5}\n"
            "  Build it with pre_process_1_data_v4.py + pre_process_2_hdf5_v4.py,\n"
            "  or pass --h5-path to an existing v4 file."
        )

    with h5py.File(h5, "r") as f:
        attrs = dict(f.attrs)
        dsets = set(f.keys())

    def _s(key):
        v = attrs.get(key, "")
        return v.decode() if isinstance(v, bytes) else str(v)

    schema = _s("schema_version")
    if schema != "v4":
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has schema_version={schema!r}, expected 'v4'.\n"
            "  v3 HDF5 files use a calendar-day maturity basis, keep exact-T=0 rows, and\n"
            "  collapse AM/PM-settled contracts -- training on one and writing to a\n"
            "  v5-named dir would mix datasets inside a single comparison table.\n"
            "  Rebuild with the _v4 preprocessing scripts, or pass --h5-path to a v4 file."
        )

    # --- v5 sample gate (schema_version alone no longer identifies the sample) ---
    # v4 and v5 share the tensor schema ("v4"); only `dataset_version` and the
    # quote-filter attrs separate the unfiltered v4 sample from the canonical
    # quote-filtered v5 one. A v4 file MUST fail here.
    dataset_version = _s("dataset_version")
    if dataset_version != "v5":
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has dataset_version={dataset_version!r}, expected 'v5'.\n"
            "  v5 is the canonical sample: v4 rows whose mid/K violates the static\n"
            "  no-arbitrage bounds (zero tolerance) are dropped -- the BS decoder cannot\n"
            "  reach them at any sigma_hat. The unfiltered v4 files carry the same\n"
            "  schema_version='v4' but are a DIFFERENT sample, retained as a robustness\n"
            "  check only and noncanonical for training.\n"
            "  Point --h5-path at deeponet_tensors_{call,put}_v5.h5."
        )

    quote_filter = _s("quote_filter")
    if quote_filter != "static_bounds_midpoint":
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has quote_filter={quote_filter!r}, expected "
            "'static_bounds_midpoint' -- the approved static no-arbitrage midpoint filter."
        )

    try:
        quote_filter_tolerance = float(attrs.get("quote_filter_tolerance", float("nan")))
    except (TypeError, ValueError):
        quote_filter_tolerance = float("nan")
    if quote_filter_tolerance != 0.0:
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has "
            f"quote_filter_tolerance={quote_filter_tolerance!r}, expected 0.0 "
            "(the approved zero-tolerance filter)."
        )

    # --- gates common to all 12 scripts (see tests/test_v5_gate_all_scripts.py) ---
    want_cp = "C" if cfg["option_type"] == "call" else "P"
    cp = _s("cp_flag")
    if cp != want_cp:
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has cp_flag={cp!r} but this script prices "
            f"{cfg['option_type']}s (expected {want_cp!r}).\n"
            "  The call and put HDF5 files are structurally identical, so pointing a call\n"
            "  script at the put file would otherwise train silently on the wrong side."
        )

    t_basis = _s("export_t_basis")
    if t_basis != "settlement":
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has export_t_basis={t_basis!r}, expected "
            "'settlement'. Maturities must be on the settlement-aware basis."
        )

    try:
        min_mat = float(attrs.get("export_min_maturity_days", float("nan")))
    except (TypeError, ValueError):
        min_mat = float("nan")
    if not abs(min_mat - 1.0) < 1e-6:
        raise SystemExit(
            f"REFUSING TO TRAIN: {h5.name} has export_min_maturity_days={min_mat!r}, "
            "expected 1.0 -- the canonical paper rule (export filter T > 1 day)."
        )

    for name in ("split_id", cfg["branch_key"]):
        if name not in dsets:
            raise SystemExit(
                f"REFUSING TO TRAIN: {h5.name} has no `{name}` dataset "
                f"(present: {sorted(dsets)})."
            )

    cfg["h5_sha256"] = _sha256_of(h5)
    cfg["dataset_version"] = dataset_version
    cfg["quote_filter"] = quote_filter
    cfg["quote_filter_tolerance"] = quote_filter_tolerance
    cfg["quote_filter_bounds"] = _s("quote_filter_bounds")
    cfg["data_provenance"] = {
        # Same four fields as the top-level cfg keys above, repeated here so a
        # reader of either location sees the same sample identity (analysis/
        # eval_to_json.py hard-fails if they ever disagree).
        "dataset_version": dataset_version,
        "quote_filter": quote_filter,
        "quote_filter_tolerance": quote_filter_tolerance,
        "quote_filter_bounds": cfg["quote_filter_bounds"],
        "schema_version": schema,
        "cp_flag": _s("cp_flag"),
        "export_min_maturity_days": _s("export_min_maturity_days"),
        "export_t_basis": _s("export_t_basis"),
        "trunk_y_columns": _s("trunk_y_columns"),
        "am_settlement_coding": _s("am_settlement_coding"),
        # Everything phase 1 recorded (t_basis, source files/hashes, row counts...).
        "phase1": {k: _s(k) for k in sorted(attrs) if k.startswith("phase1_")},
    }
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO on puts with spot history input (V3, analytical BS)")
    parser.add_argument("--epochs-adam", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--head-hidden", type=int, default=None)
    parser.add_argument("--h5-path", type=str, default=None,
                        help="Override the input HDF5 (default: deeponet_tensors_put_v5.h5). "
                             "Must be a dataset_version=v5 (quote-filtered) file; v4 (unfiltered) and v3 "
                             "files are refused at startup "
                             "(different maturity basis / filtering / contract identity). "
                             "See _vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override the output directory (default: results_spot_history_v5). "
                             "Never point this at a v3/v4 results_* dir -- those weights back the "
                             "current report table and must not be overwritten.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best_model.pth and run test eval + diagnostics only.")
    parser.add_argument("--stable-log", action="store_true",
                        help="Price log(V/K) via stable log_ndtr instead of log(clamp(price, 1e-8)). "
                             "Safe for eval; for training the gradient near ATM explodes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()

    if args.epochs_adam is not None:
        cfg["epochs_adam"] = args.epochs_adam
    if args.epochs_finetune is not None:
        cfg["epochs_finetune"] = args.epochs_finetune
    if args.max_train_batches is not None:
        cfg["max_train_batches"] = args.max_train_batches
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.finetune_lr is not None:
        cfg["finetune_lr"] = args.finetune_lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.latent_dim is not None:
        cfg["latent_dim"] = args.latent_dim
    if args.head_hidden is not None:
        cfg["head_hidden"] = args.head_hidden
    if args.h5_path is not None:
        cfg["h5_path"] = str(Path(args.h5_path))
    if args.results_dir is not None:
        cfg["results_dir"] = str(Path(args.results_dir))
    if args.eval_only:
        cfg["eval_only"] = True
    if args.stable_log:
        cfg["use_stable_log"] = True
    add_provenance(cfg)

    print("Config:", json.dumps(cfg, indent=2))
    train_model(cfg)
