"""
Train FNO + MLP model on put options using VIX OHLC history (vix_history).

Architecture:
  Network 1: 2D FNO on VIX history grid (21x4) -> sigma_hat (positive scalar)
  Network 2: MLP([log_moneyness, T_years, r, q, sigma_hat]) -> log(V/K)

Losses:
  - Data MSE on log(V/K)
  - Black-Scholes PDE residual in normalized price space
  - Option-type-specific arbitrage constraints (negative delta, positive gamma for puts)
  - BS analytical anchor (Network 2 only, synthetic samples)
"""

import argparse
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
# NETWORK 1: FNO VOLATILITY ESTIMATOR
# ==========================================

class FNO_VolEstimator(nn.Module):
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.width = width

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
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.grid_h, self.grid_w)
        x = self.fc0(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        x = x.view(x.shape[0], -1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return F.softplus(x)


# ==========================================
# NETWORK 2: MLP PRICE PREDICTOR
# ==========================================

class MLP_Predictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=4, output_dim=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ==========================================
# COMBINED MODEL
# ==========================================

class TwoNetworkModel(nn.Module):
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32, hidden_dim=128, num_layers=4):
        super().__init__()
        self.net1 = FNO_VolEstimator(grid_h, grid_w, modes1, modes2, width)
        self.net2 = MLP_Predictor(input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)

    def forward(self, branch, log_moneyness, T_years, r, q):
        sigma_hat = self.net1(branch)
        nn2_input = torch.cat([log_moneyness, T_years, r, q, sigma_hat], dim=1)
        log_v_hat = self.net2(nn2_input)
        return sigma_hat, log_v_hat

    def forward_with_sigma(self, log_moneyness, T_years, r, q, sigma):
        nn2_input = torch.cat([log_moneyness, T_years, r, q, sigma], dim=1)
        return self.net2(nn2_input)


# ==========================================
# BLACK-SCHOLES ANALYTICAL PRICES
# ==========================================

def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


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

    return torch.clamp(price, min=1e-10)


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def compute_derivatives(log_v_hat, log_moneyness, T_years):
    v = torch.exp(log_v_hat)
    dv_dlogM = torch.autograd.grad(
        v, log_moneyness, grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True
    )[0]
    d2v_dlogM2 = torch.autograd.grad(
        dv_dlogM, log_moneyness, grad_outputs=torch.ones_like(dv_dlogM),
        create_graph=True, retain_graph=True
    )[0]
    dv_dT = torch.autograd.grad(
        v, T_years, grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True
    )[0]
    return v, dv_dlogM, d2v_dlogM2, dv_dT


def pde_residual(v, dv_dlogM, d2v_dlogM2, dv_dT, log_moneyness, sigma_hat, r, q):
    M = torch.exp(log_moneyness)
    M_safe = torch.clamp(M, min=1e-10)
    dv_dM = dv_dlogM / M_safe
    d2v_dM2 = (d2v_dlogM2 - dv_dlogM) / (M_safe ** 2)
    residual = (
        -dv_dT
        + 0.5 * (sigma_hat ** 2) * (M ** 2) * d2v_dM2
        + (r - q) * M * dv_dM
        - r * v
    )
    return residual


def compute_loss(model, branch, log_moneyness, T_years, r, q, target_v_log,
                 lambdas, option_type, device, bs_anchor_batch_size=256):
    sigma_hat, log_v_hat = model(branch, log_moneyness, T_years, r, q)

    loss_data = F.mse_loss(log_v_hat, target_v_log)

    v, dv_dlogM, d2v_dlogM2, dv_dT = compute_derivatives(log_v_hat, log_moneyness, T_years)
    residual = pde_residual(v, dv_dlogM, d2v_dlogM2, dv_dT, log_moneyness, sigma_hat, r, q)
    loss_pde = torch.mean(residual ** 2)

    M_safe = torch.clamp(torch.exp(log_moneyness), min=1e-10)
    dv_dM = dv_dlogM / M_safe
    d2v_dM2 = (d2v_dlogM2 - dv_dlogM) / (M_safe ** 2)

    if option_type == "call":
        loss_arb = torch.mean(F.relu(-dv_dM)) + torch.mean(F.relu(-d2v_dM2))
    else:
        loss_arb = torch.mean(F.relu(dv_dM)) + torch.mean(F.relu(-d2v_dM2))

    bs_log_m = torch.empty(bs_anchor_batch_size, 1, device=device).uniform_(-1.5, 1.5)
    bs_T = torch.empty(bs_anchor_batch_size, 1, device=device).uniform_(1.0 / 365.0, 5.0)
    bs_r = torch.empty(bs_anchor_batch_size, 1, device=device).uniform_(0.0, 0.10)
    bs_q = torch.empty(bs_anchor_batch_size, 1, device=device).uniform_(0.0, 0.05)
    bs_sigma = torch.empty(bs_anchor_batch_size, 1, device=device).uniform_(0.05, 1.00)

    bs_price = bs_normalized_price(bs_log_m, bs_T, bs_r, bs_q, bs_sigma, option_type)
    bs_target = torch.log(bs_price)

    bs_log_v_hat = model.forward_with_sigma(bs_log_m, bs_T, bs_r, bs_q, bs_sigma)
    loss_bs = F.mse_loss(bs_log_v_hat, bs_target)

    loss_total = (
        lambdas["data"] * loss_data
        + lambdas["pde"] * loss_pde
        + lambdas["arb"] * loss_arb
        + lambdas["bs_anchor"] * loss_bs
    )

    loss_dict = {
        "data": loss_data.item(),
        "pde": loss_pde.item(),
        "arb": loss_arb.item(),
        "bs_anchor": loss_bs.item(),
        "total": loss_total.item(),
    }
    return loss_total, loss_dict


# ==========================================
# DATASET
# ==========================================

class H5BranchDataset(Dataset):
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
            trunk_y[:, 0:1],
            trunk_y[:, 1:2],
            trunk_y[:, 2:3],
            trunk_y[:, 3:4],
            target,
        )
    return DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=shuffle)


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
    bs_anchor_batch_size = config["bs_anchor_batch_size"]
    max_train_batches = config.get("max_train_batches", None)
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    fno_width = config["fno_width"]

    lambdas = {
        "data": config["lambda_data"],
        "pde": config["lambda_pde"],
        "arb": config["lambda_arb"],
        "bs_anchor": config["lambda_bs_anchor"],
    }

    h5_path = config["h5_path"]
    branch_key = config["branch_key"]
    grid_h = config["grid_h"]
    grid_w = config["grid_w"]
    modes1 = config["modes1"]
    modes2 = config["modes2"]
    option_type = config["option_type"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

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

    model = TwoNetworkModel(grid_h, grid_w, modes1, modes2, fno_width, hidden_dim, num_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"FNO-DeepONet Model initialized with hidden_dim={hidden_dim}, "
        f"num_layers={num_layers}, fno_width={fno_width}"
    )
    print(f"Model parameters: {total_params:,}")

    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer_adam, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    all_train_losses = []
    all_val_losses = []

    # ============ PHASE 1: ADAM ============
    print("\n" + "=" * 60)
    print("PHASE 1: Adam Optimizer (FNO Parameter Groups)")
    print("=" * 60)

    epochs_no_improve = 0
    phase1_epochs = 0

    for epoch in range(epochs_adam):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        comps = {"data": 0.0, "pde": 0.0, "arb": 0.0, "bs_anchor": 0.0}
        n_batches_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            n_batches_run += 1

            branch, log_m, T, r, q, target = [b.to(device) for b in batch]
            log_m = log_m.clone().requires_grad_(True)
            T = T.clone().requires_grad_(True)

            optimizer_adam.zero_grad()
            loss, loss_dict = compute_loss(
                model, branch, log_m, T, r, q, target, lambdas, option_type, device,
                bs_anchor_batch_size,
            )
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
        print(
            f"  Components - Data: {comps['data']:.4f} | PDE: {comps['pde']:.4f} | "
            f"Arb: {comps['arb']:.4f} | BS: {comps['bs_anchor']:.4f}"
        )
        print(f"  Epochs without improvement: {epochs_no_improve}/{patience_adam}")
        print("-" * 60)

        if epochs_no_improve >= patience_adam:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

        torch.cuda.empty_cache()

    print(f"\n*** Adam Phase Complete. Best Val MSE: {best_val:.6f} ***")

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

    for epoch in range(epochs_finetune):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        comps = {"data": 0.0, "pde": 0.0, "arb": 0.0, "bs_anchor": 0.0}
        n_batches_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            n_batches_run += 1

            branch, log_m, T, r, q, target = [b.to(device) for b in batch]
            log_m = log_m.clone().requires_grad_(True)
            T = T.clone().requires_grad_(True)

            optimizer_ft.zero_grad()
            loss, loss_dict = compute_loss(
                model, branch, log_m, T, r, q, target, lambdas, option_type, device,
                bs_anchor_batch_size,
            )
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

    test_mse_log /= total_samples
    test_rmse_log = np.sqrt(test_mse_log)
    test_mse_price /= total_samples
    test_rmse_price = np.sqrt(test_mse_price)
    test_mae_price /= total_samples

    all_log_targets = torch.cat(all_log_targets).numpy().ravel()
    all_log_preds = torch.cat(all_log_preds).numpy().ravel()
    ss_res = np.sum((all_log_targets - all_log_preds) ** 2)
    ss_tot = np.sum((all_log_targets - np.mean(all_log_targets)) ** 2)
    r2_log = 1.0 - ss_res / (ss_tot + 1e-10)

    all_price_targets = np.exp(all_log_targets)
    all_price_preds = np.exp(all_log_preds)
    ss_res_p = np.sum((all_price_targets - all_price_preds) ** 2)
    ss_tot_p = np.sum((all_price_targets - np.mean(all_price_targets)) ** 2)
    r2_price = 1.0 - ss_res_p / (ss_tot_p + 1e-10)

    print(f"Final Test MSE (log):  {test_mse_log:.6f}")
    print(f"Test MSE (log):        {test_mse_log:.8f}")
    print(f"Test RMSE (log):       {test_rmse_log:.6f}")
    print(f"Test MSE (price):      {test_mse_price:.8f}")
    print(f"Test RMSE (price):     {test_rmse_price:.6f}")
    print(f"Test MAE (price):      {test_mae_price:.6f}")
    print(f"Test R^2 (log):        {r2_log:.6f}")
    print(f"Test R^2 (price):      {r2_price:.6f}")

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
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    h5_path = str(project_root / "wrds_data_2020-2025" / "deeponet_tensors_put.h5")

    return {
        "seed": 42,
        "epochs_adam": 100,
        "epochs_finetune": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "finetune_lr": 1e-5,
        "grad_clip": 1.0,
        "patience_adam": 3,
        "patience_finetune": 3,
        "min_finetune_epochs": 10,
        "bs_anchor_batch_size": 256,
        "max_train_batches": None,
        "hidden_dim": 128,
        "num_layers": 4,
        "fno_width": 32,
        "lambda_data": 1.0,
        "lambda_pde": 0.5,
        "lambda_arb": 0.5,
        "lambda_bs_anchor": 0.5,
        "h5_path": h5_path,
        "branch_key": "vix_history",
        "grid_h": 21,
        "grid_w": 4,
        "modes1": 6,
        "modes2": 2,
        "option_type": "put",
        "results_dir": str(script_dir / "results_vix_history"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO+MLP on puts with VIX history input")
    parser.add_argument("--epochs-adam", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
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

    print("Config:", json.dumps(cfg, indent=2))
    train_model(cfg)
