"""
Train FNO model on put options using implied volatility surface (branch_u).
V3: Network 2 removed; analytical Black-Scholes used directly.

Architecture:
  Network 1: 2D FNO on IV surface grid (11x17) -> sigma_hat (positive scalar)
  Pricing:   bs_normalized_price([log_moneyness, T_years, r, q, sigma_hat]) -> V/K

Loss: Data MSE on log(V/K). No PDE / arbitrage / BS-anchor terms.
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
# NETWORK 1: FNO VOLATILITY ESTIMATOR (SiLU)
# ==========================================

class FNO_VolEstimator(nn.Module):
    """FNO-based market-state-only volatility estimator.

    Takes a 2D grid, passes through 4 FNO blocks with residual connections,
    then an MLP projection to a positive scalar sigma_hat.
    Uses SiLU activation throughout.
    """
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
        x = self.fc3(x)
        return F.softplus(x)


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


# ==========================================
# COMBINED MODEL
# ==========================================

class VolEstimatorModel(nn.Module):
    def __init__(self, grid_h, grid_w, modes1, modes2, width=32, option_type="call"):
        super().__init__()
        self.net1 = FNO_VolEstimator(grid_h, grid_w, modes1, modes2, width)
        self.option_type = option_type

    def forward(self, branch, log_moneyness, T_years, r, q):
        sigma_hat = self.net1(branch)
        price = bs_normalized_price(log_moneyness, T_years, r, q, sigma_hat, self.option_type)
        log_v_hat = torch.log(torch.clamp(price, min=1e-8))
        return sigma_hat, log_v_hat


# ==========================================
# TRAINING LOSS (data-only)
# ==========================================

def compute_loss(model, branch, log_moneyness, T_years, r, q, target_v_log):
    sigma_hat, log_v_hat = model(branch, log_moneyness, T_years, r, q)
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

    # Save config
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

    model = VolEstimatorModel(grid_h, grid_w, modes1, modes2, fno_width, option_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FNO_VolEstimator initialized with fno_width={fno_width}")
    print(f"Total model parameters: {total_params:,}")

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

    for epoch in range(epochs_adam):
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

    log_errs = (all_log_preds - all_log_targets) ** 2
    print(f"log-error percentiles (squared):")
    for p in [50, 90, 99, 99.9, 100]:
        print(f"  p{p}: {np.percentile(log_errs, p):.6f}")
    print(f"#samples with sq.err > 10: {(log_errs > 10.0).sum()} / {len(log_errs)}")
    print(f"clamp-hit count (pred <= log(1.0001e-8)): {(all_log_preds < -18.42).sum()}")

    print(f"Final Test MSE (log):  {test_mse_log:.6f}")
    print(f"Test MSE (log):        {test_mse_log:.8f}")
    print(f"Test RMSE (log):       {test_rmse_log:.6f}")
    print(f"Test MSE (price):      {test_mse_price:.8f}")
    print(f"Test RMSE (price):     {test_rmse_price:.6f}")
    print(f"Test MAE (price):      {test_mae_price:.6f}")
    print(f"Test R^2 (log):        {r2_log:.6f}")
    print(f"Test R^2 (price):      {r2_price:.6f}")

    # ============ SAVE OUTPUTS ============
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
    """Return the default configuration for this script."""
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
        "patience_adam": 10,
        "patience_finetune": 10,
        "min_finetune_epochs": 10,
        "max_train_batches": None,
        "fno_width": 32,
        "h5_path": h5_path,
        "branch_key": "branch_u",
        "grid_h": 11,
        "grid_w": 17,
        "modes1": 4,
        "modes2": 8,
        "option_type": "put",
        "results_dir": str(script_dir / "results_vol_surface"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO on puts with vol surface input (V3, analytical BS)")
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
