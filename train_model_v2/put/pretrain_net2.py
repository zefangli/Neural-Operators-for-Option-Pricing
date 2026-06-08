"""
Pre-train Network 2 (MLP_Predictor) to approximate the Black-Scholes analytical
normalized put price formula to very low error.

Architecture:
  MLP([log_moneyness, T_years, r, q, sigma]) -> log(V/K)  (unconstrained)

Loss:
  MSE between predicted log(V/K) and analytical log(BS_price / K)

Training:
  - Pure supervised learning on synthetic data sampled from ranges
    that cover the empirical market data (read from HDF5 once).
  - Single loss, no PDE/arbitrage/BS-anchor terms.
  - Target: test MSE < 1e-8 on held-out synthetic samples.
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
# ==========================================
# NETWORK 2: MLP PRICE PREDICTOR (SiLU)
# ==========================================

class MLP_Predictor(nn.Module):
    """MLP taking [log_moneyness, T_years, r, q, sigma] -> unconstrained log(V/K).

    Input-normalization stats live as buffers on the model so they travel with
    state_dict. Callers must NOT normalize inputs externally.
    """
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=4, output_dim=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def set_input_stats(self, mean, std):
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_std.copy_(std.to(self.input_std.device))

    def forward(self, x):
        return self.net((x - self.input_mean) / self.input_std)


# ==========================================
# BLACK-SCHOLES ANALYTICAL PRICES
# ==========================================

def _norm_cdf(x):
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    return 0.5 * (1.0 + torch.erf(x / sqrt2))


def bs_normalized_price(log_moneyness, T_years, r, q, sigma, option_type="put"):
    M = torch.exp(log_moneyness)
    sqrt_T = torch.sqrt(torch.clamp(T_years, min=1e-10))
    sigma_safe = torch.clamp(sigma, min=1e-10)

    d1 = (log_moneyness + (r - q + 0.5 * sigma_safe ** 2) * T_years) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    if option_type == "call":
        price = M * torch.exp(-q * T_years) * _norm_cdf(d1) - torch.exp(-r * T_years) * _norm_cdf(d2)
    else:
        price = torch.exp(-r * T_years) * _norm_cdf(-d2) - M * torch.exp(-q * T_years) * _norm_cdf(-d1)

    return price  # unclamped; caller filters tiny prices before log()


# ==========================================
# SYNTHETIC DATASET
# ==========================================

class SyntheticBSDataset(Dataset):
    """Generates synthetic (log_moneyness, T_years, r, q, sigma) -> log(V/K) pairs.

    Ranges are computed from empirical HDF5 data to ensure good coverage.
    """
    def __init__(self, h5_path, n_samples, option_type, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.option_type = option_type
        self.rng = np.random.RandomState(seed)

        # Compute empirical ranges from HDF5 training split
        print(f"Reading empirical ranges from {h5_path} (train split)...")
        with h5py.File(h5_path, "r") as f:
            split_ids = f["split_id"][:]
            train_mask = split_ids == 0
            trunk = f["trunk_y"][train_mask]

        log_m = trunk[:, 0]
        T = trunk[:, 1]
        r = trunk[:, 2]
        q = trunk[:, 3]

        # Buffered ranges
        log_m_min, log_m_max = float(log_m.min()), float(log_m.max())
        T_min, T_max = float(T.min()), float(T.max())
        r_min, r_max = float(r.min()), float(r.max())
        q_min, q_max = float(q.min()), float(q.max())

        self.log_m_range = (log_m_min - 0.2, log_m_max + 0.2)
        # Tightened box: minimum 1 week to expiry, sigma in [0.10, 0.80].
        # These cut the steepest regions of log(BS) where MLP precision suffers.
        self.T_range = (max(7.0 / 365.0, T_min - 0.1), T_max + 0.5)
        self.r_range = (r_min - 0.01, r_max + 0.01)
        self.q_range = (q_min - 0.01, q_max + 0.01)
        self.sigma_range = (0.10, 0.80)

        print(f"  Empirical: log_m [{log_m_min:.4f}, {log_m_max:.4f}], "
              f"T [{T_min:.4f}, {T_max:.4f}], r [{r_min:.6f}, {r_max:.6f}], "
              f"q [{q_min:.6f}, {q_max:.6f}]")
        print(f"  Synthetic: log_m {self.log_m_range}, T {self.T_range}, "
              f"r {self.r_range}, q {self.q_range}, sigma {self.sigma_range}")

        # Pre-generate all samples
        self._generate()

    def _generate(self):
        target_n = self.n_samples
        # Tightened filter: keep only samples whose BS price is comfortably
        # above the numerical floor, so log() targets are well-conditioned.
        MIN_PRICE = 1e-4
        chunks_inp = []
        chunks_tgt = []
        collected = 0
        total_attempted = 0

        while collected < target_n:
            n = max(target_n * 2, (target_n - collected) * 2, 1024)
            # Use float64 throughout so BS targets are precise to ~1e-15.
            log_m = self.rng.uniform(*self.log_m_range, n).astype(np.float64)
            T = self.rng.uniform(*self.T_range, n).astype(np.float64)
            r = self.rng.uniform(*self.r_range, n).astype(np.float64)
            q = self.rng.uniform(*self.q_range, n).astype(np.float64)
            sigma = self.rng.uniform(*self.sigma_range, n).astype(np.float64)
            inputs = np.column_stack([log_m, T, r, q, sigma])

            inp_t = torch.from_numpy(inputs)
            price = bs_normalized_price(
                inp_t[:, 0:1], inp_t[:, 1:2], inp_t[:, 2:3],
                inp_t[:, 3:4], inp_t[:, 4:5], self.option_type,
            ).squeeze(-1).numpy()

            mask = price > MIN_PRICE
            kept_inputs = inputs[mask]
            kept_prices = price[mask]

            chunks_inp.append(kept_inputs)
            chunks_tgt.append(np.log(kept_prices).astype(np.float64).reshape(-1, 1))
            collected += kept_inputs.shape[0]
            total_attempted += n

        self.inputs = np.concatenate(chunks_inp, axis=0)[:target_n]
        self.targets = np.concatenate(chunks_tgt, axis=0)[:target_n]

        accept_rate = collected / max(total_attempted, 1)
        print(f"  Generated {target_n:,} samples (acceptance {accept_rate:.2%} "
              f"after filter price > {MIN_PRICE:.0e}); "
              f"target log-price range "
              f"[{self.targets.min():.3f}, {self.targets.max():.3f}]")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx]),
        )


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

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    n_train = config["n_train_samples"]
    n_val = config["n_val_samples"]
    option_type = config["option_type"]
    h5_path = config["h5_path"]
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create datasets
    print("\nCreating synthetic datasets...")
    train_ds = SyntheticBSDataset(h5_path, n_train, option_type, seed=seed)
    val_ds = SyntheticBSDataset(h5_path, n_val, option_type, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size * 2, shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model — float64 throughout to remove fp32 noise floor (~1e-6 in MSE).
    model = MLP_Predictor(input_dim=5, hidden_dim=hidden_dim,
                          num_layers=num_layers, output_dim=1).double().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MLP_Predictor initialized: hidden_dim={hidden_dim}, "
          f"num_layers={num_layers}, params={total_params:,} (float64)")

    # Compute & freeze input normalization stats from training data
    input_mean = torch.from_numpy(
        train_ds.inputs.mean(axis=0).astype(np.float64)
    ).to(device)
    input_std = torch.from_numpy(
        np.clip(train_ds.inputs.std(axis=0), 1e-8, None).astype(np.float64)
    ).to(device)
    model.set_input_stats(input_mean, input_std)
    print(f"Input normalization stats:")
    print(f"  mean = {input_mean.cpu().numpy()}")
    print(f"  std  = {input_std.cpu().numpy()}")

    # weight_decay=0 to remove the L2 floor that prevents loss from going to 0.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    all_train_losses = []
    all_val_losses = []
    epochs_no_improve = 0
    patience = config.get("patience", 50)
    skip_adam = config.get("skip_adam", False)

    # ----- Optional resume: skip Adam entirely, reload existing checkpoint
    if skip_adam:
        ckpt_path = results_dir / "net2_pretrained.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--skip-adam requires {ckpt_path} to exist."
            )
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device)
        )
        print(f"\n[skip-adam] Loaded checkpoint from {ckpt_path}")

        # Restore previous loss history (so the rewritten file is continuous)
        hist_path = results_dir / "loss_history.txt"
        if hist_path.exists():
            with open(hist_path) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3 and parts[0].isdigit():
                        try:
                            all_train_losses.append(float(parts[1]))
                            all_val_losses.append(float(parts[2]))
                        except ValueError:
                            pass
            if all_val_losses:
                best_val_loss = min(all_val_losses)
            print(f"[skip-adam] Restored {len(all_train_losses)} prior entries; "
                  f"best_val_loss = {best_val_loss:.3e}")

    print("\n" + "=" * 60)
    print("PRE-TRAINING NETWORK 2 ON BS ANALYTICAL FORMULA")
    print("=" * 60)

    for epoch in (range(0) if skip_adam else range(epochs)):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            log_v_hat = model(inputs)
            loss = F.mse_loss(log_v_hat, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.3e}")

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                log_v_hat = model(inputs)
                val_loss += F.mse_loss(log_v_hat, targets, reduction="sum").item()
                val_batches += 1
        val_loss /= len(val_ds)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        all_train_losses.append(avg_train_loss)
        all_val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(results_dir / "net2_pretrained.pth"))
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"\nEpoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.2e}")
        print(f"  Train MSE: {avg_train_loss:.3e} | "
              f"Val MSE: {val_loss:.3e}{improved}")
        print(f"  Epochs without improvement: {epochs_no_improve}/{patience}")
        print(f"  Target: < 1e-8 | Current RMSE: {np.sqrt(val_loss):.3e}")
        print("-" * 60)

        if epochs_no_improve >= patience:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

        torch.cuda.empty_cache()

    if not skip_adam:
        print(f"\n*** Adam phase complete. Best Val MSE so far: {best_val_loss:.3e} ***")
    else:
        print(f"\n*** Skipped Adam; resuming with best Val MSE: {best_val_loss:.3e} ***")
    adam_epoch_count = len(all_train_losses)

    # ==========================================
    # L-BFGS FINE-TUNING PHASE
    # ==========================================
    lbfgs_iters = config.get("lbfgs_iters", 100)
    if lbfgs_iters > 0:
        print("\n" + "=" * 60)
        print(f"L-BFGS FINE-TUNING ({lbfgs_iters} iters, full-batch via chunks)")
        print("=" * 60)

        # Restart from best Adam checkpoint
        model.load_state_dict(torch.load(
            str(results_dir / "net2_pretrained.pth"), map_location=device
        ))

        # Materialize full train/val tensors on device in float64
        X_train = torch.from_numpy(train_ds.inputs).to(device, dtype=torch.float64)
        y_train = torch.from_numpy(train_ds.targets).to(device, dtype=torch.float64)
        X_val = torch.from_numpy(val_ds.inputs).to(device, dtype=torch.float64)
        y_val = torch.from_numpy(val_ds.targets).to(device, dtype=torch.float64)
        n_train_tot = X_train.shape[0]
        lbfgs_chunk = config.get("lbfgs_chunk_size", 50000)

        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
        )

        log_every = max(1, lbfgs_iters // 20)
        lbfgs_start = time.time()
        for it in range(lbfgs_iters):
            iter_start = time.time()
            model.train()

            def closure():
                lbfgs.zero_grad()
                total = 0.0
                for i in range(0, n_train_tot, lbfgs_chunk):
                    pred = model(X_train[i:i + lbfgs_chunk])
                    li = F.mse_loss(
                        pred, y_train[i:i + lbfgs_chunk], reduction="sum"
                    ) / n_train_tot
                    li.backward()
                    total += li.item()
                return torch.tensor(total, device=device, dtype=torch.float64)

            train_loss = float(lbfgs.step(closure))
            iter_time = time.time() - iter_start

            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(X_val), y_val).item()

            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           str(results_dir / "net2_pretrained.pth"))
                improved = " *** New Best ***"

            if (it + 1) % log_every == 0 or it == 0 or it == lbfgs_iters - 1:
                elapsed = time.time() - lbfgs_start
                avg_s_per_iter = elapsed / (it + 1)
                eta_s = avg_s_per_iter * (lbfgs_iters - it - 1)
                print(f"L-BFGS {it+1:4d}/{lbfgs_iters} | "
                      f"Train MSE: {train_loss:.3e} | "
                      f"Val MSE: {val_loss:.3e}{improved} | "
                      f"iter {iter_time:.1f}s | "
                      f"avg {avg_s_per_iter:.1f}s/it | "
                      f"ETA {eta_s/60:.1f}min")

        # Free L-BFGS tensors
        del X_train, y_train, X_val, y_val
        torch.cuda.empty_cache()

        print(f"\n*** L-BFGS phase complete. Best Val MSE: {best_val_loss:.3e} ***")

    # Save final model (post-LBFGS if it ran, otherwise post-Adam)
    torch.save(model.state_dict(), str(results_dir / "net2_final.pth"))

    # Evaluate best model on validation set
    print("\nEvaluating best model on validation set...")
    model.load_state_dict(torch.load(str(results_dir / "net2_pretrained.pth"),
                                     map_location=device))
    model.eval()
    final_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            log_v_hat = model(inputs)
            final_val_loss += F.mse_loss(log_v_hat, targets, reduction="sum").item()
    final_val_loss /= len(val_ds)
    final_rmse = np.sqrt(final_val_loss)

    print(f"  Best model val MSE:  {final_val_loss:.3e}")
    print(f"  Best model val RMSE: {final_rmse:.3e}")

    # Save loss history
    with open(results_dir / "loss_history.txt", "w") as f:
        f.write("Epoch\tTrain_MSE\tVal_MSE\tVal_RMSE\n")
        for i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
            f.write(f"{i+1}\t{tl:.3e}\t{vl:.3e}\t{np.sqrt(vl):.3e}\n")
        f.write(f"\n--- Final Validation Metrics ---\n")
        f.write(f"MSE:  {final_val_loss:.3e}\n")
        f.write(f"RMSE: {final_rmse:.3e}\n")

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label="Train MSE")
    plt.plot(all_val_losses, label="Val MSE")
    plt.axhline(y=1e-8, color="r", linestyle="--", alpha=0.5, label="Target (1e-8)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Network 2 Pre-training (Best Val MSE: {final_val_loss:.3e})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_plot.png")
    plt.close()

    print(f"\nResults saved to {results_dir}")
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
        "epochs": 500,
        "batch_size": 1024,
        "lr": 1e-3,
        "hidden_dim": 128,
        "num_layers": 4,
        "n_train_samples": 500000,
        "n_val_samples": 100000,
        "patience": 50,
        "lbfgs_iters": 100,
        "lbfgs_chunk_size": 50000,
        "option_type": "put",
        "h5_path": h5_path,
        "results_dir": str(script_dir / "pretrained_net2"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-train Network 2 on BS analytical formula (puts)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--target-n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lbfgs-iters", type=int, default=None,
                        help="Number of L-BFGS iters after Adam (0 to skip).")
    parser.add_argument("--lbfgs-chunk-size", type=int, default=None)
    parser.add_argument("--skip-adam", action="store_true",
                        help="Skip Adam; reload net2_pretrained.pth and "
                             "continue with L-BFGS only.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        cfg["num_layers"] = args.num_layers
    if args.target_n_samples is not None:
        cfg["n_train_samples"] = args.target_n_samples
        cfg["n_val_samples"] = max(10000, args.target_n_samples // 5)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lbfgs_iters is not None:
        cfg["lbfgs_iters"] = args.lbfgs_iters
    if args.lbfgs_chunk_size is not None:
        cfg["lbfgs_chunk_size"] = args.lbfgs_chunk_size
    if args.skip_adam:
        cfg["skip_adam"] = True

    print("Config:", json.dumps(cfg, indent=2))
    train_model(cfg)
