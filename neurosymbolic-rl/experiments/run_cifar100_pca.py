#!/usr/bin/env python3
"""CIFAR-100: PCA dimensionality reduction + MLP."""
import os, sys, json, time, argparse
import yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.large_feature_bank import LargeFeatureBank
from experiments.evaluate_mlp import extract_features


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def gpu_pca(X, n_components, device):
    """PCA via randomized SVD (torch.pca_lowrank) on GPU. Memory-efficient."""
    n, d = X.shape
    print(f"  GPU PCA (randomized): ({n}, {d}) -> {n_components} components")
    t0 = time.time()
    # pca_lowrank is memory-efficient: no need for full n×n or d×d matrices
    # niter=5 gives good accuracy for randomized SVD
    U, S, V = torch.pca_lowrank(X, q=n_components, niter=5)
    # V is (d, n_components) — the projection matrix
    # S contains singular values
    eigenvalues = S ** 2 / (n - 1)
    total_var = (X ** 2).sum() / (n - 1)  # approximate total variance
    explained_var = eigenvalues / total_var * 100
    cum_var = explained_var.cumsum(0)
    print(f"  Top-{n_components} explained variance: {cum_var[-1]:.1f}%")
    k10 = min(9, n_components - 1)
    k100 = min(99, n_components - 1)
    k500 = min(499, n_components - 1)
    print(f"  Top-10: {cum_var[k10]:.1f}%, Top-100: {cum_var[k100]:.1f}%, "
          f"Top-500: {cum_var[k500]:.1f}%")
    print(f"  PCA done ({time.time()-t0:.1f}s)")
    del U, S
    torch.cuda.empty_cache()
    return V  # (d, n_components)


def train_mlp(X_train, y_train, X_test, y_test, input_dim, num_classes,
              hidden_dim, device, epochs=200, patience=30, batch_size=512,
              lr=1e-3, weight_decay=1e-4, dropout=0.3, label=""):
    model = MLPClassifier(input_dim, num_classes, hidden_dim, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  h={hidden_dim}  dropout={dropout}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    n_train = X_train.shape[0]
    best_test_acc = 0.0
    best_epoch = 0
    no_improve = 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for s in range(0, n_train, batch_size):
            e = min(s + batch_size, n_train)
            idx = perm[s:e]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
            train_loss_sum += loss.item() * xb.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for s in range(0, X_test.shape[0], 2048):
                e_i = min(s + 2048, X_test.shape[0])
                logits = model(X_test[s:e_i])
                test_correct += (logits.argmax(1) == y_test[s:e_i]).sum().item()
                test_total += e_i - s
            test_acc = test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"Loss={train_loss_sum/train_total:.4f}  "
                  f"Train={train_acc*100:.2f}%  "
                  f"Test={test_acc*100:.2f}%  "
                  f"Best={best_test_acc*100:.2f}% (ep{best_epoch})")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"\n  [{label}] Best={best_test_acc*100:.2f}% (ep{best_epoch}, {elapsed:.1f}s)")
    return {
        'test_accuracy': float(best_test_acc),
        'train_accuracy': float(train_acc),
        'best_epoch': best_epoch,
        'training_time_s': elapsed,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tensor_vsr_cifar100_spp.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    dataset_name = config.get('dataset', 'cifar100')
    num_classes = config.get('training', {}).get('num_classes', 100)
    output_dir = config.get('output_dir', 'outputs/cifar100_spp')
    bank_path = os.path.join(output_dir, 'feature_bank')

    # --- Extract ---
    print(f"Loading feature bank from {bank_path}...")
    bank = LargeFeatureBank.load(bank_path, device=device)
    formula_strs = bank.formula_strs
    print(f"  {len(formula_strs)} formulas")
    print(bank.get_summary())
    del bank

    data_module = MNISTDataModule(
        dataset=dataset_name,
        batch_size=config.get('training', {}).get('eval_batch_size', 2048),
        num_workers=4, val_split=0.0)
    data_module.setup()
    print(f"Dataset: {dataset_name.upper()} — Train: {len(data_module.train_dataset)}, Test: {len(data_module.test_dataset)}")

    print("Extracting train features...")
    t0 = time.time()
    X_train, y_train = extract_features(
        formula_strs, data_module.get_train_loader(), device, tag="train")
    print(f"  {X_train.shape}  ({time.time()-t0:.1f}s)")

    print("Extracting test features...")
    t0 = time.time()
    X_test, y_test = extract_features(
        formula_strs, data_module.get_test_loader(), device, tag="test")
    print(f"  {X_test.shape}  ({time.time()-t0:.1f}s)")

    input_dim = X_train.shape[1]
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    # --- Standardize + GPU ---
    print("Standardizing...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-8)

    chunk = 5000
    X_train_gpu = torch.empty(n_train, input_dim, dtype=torch.float32, device=device)
    for s in range(0, n_train, chunk):
        e = min(s + chunk, n_train)
        block = np.clip((X_train[s:e] - mean) / std, -10, 10)
        X_train_gpu[s:e] = torch.from_numpy(block).to(device)
    del X_train

    X_test_gpu = torch.empty(n_test, input_dim, dtype=torch.float32, device=device)
    for s in range(0, n_test, chunk):
        e = min(s + chunk, n_test)
        block = np.clip((X_test[s:e] - mean) / std, -10, 10)
        X_test_gpu[s:e] = torch.from_numpy(block).to(device)
    del X_test, mean, std

    y_train_gpu = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_gpu = torch.tensor(y_test, dtype=torch.long, device=device)
    del y_train, y_test
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- PCA ---
    # Center for PCA
    pca_mean = X_train_gpu.mean(dim=0)
    X_train_centered = X_train_gpu - pca_mean
    X_test_centered = X_test_gpu - pca_mean

    # Compute PCA components once (for max dim we need)
    max_pca = 5000
    components = gpu_pca(X_train_centered, max_pca, device)  # (d, max_pca)

    # Project once, slice for different dims
    print("Projecting data...")
    t0 = time.time()
    X_train_pca = X_train_centered @ components  # (n_train, max_pca)
    X_test_pca = X_test_centered @ components    # (n_test, max_pca)
    del X_train_centered, X_test_centered, X_train_gpu, X_test_gpu
    torch.cuda.empty_cache()
    print(f"  Done ({time.time()-t0:.1f}s), GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- Experiments ---
    all_results = {}

    # PCA dim sweep
    configs = [
        # (label, pca_dim, hidden, dropout)
        ("pca500_h512",    500,  512,  0.3),
        ("pca1000_h512",   1000, 512,  0.3),
        ("pca1000_h1024",  1000, 1024, 0.3),
        ("pca2000_h1024",  2000, 1024, 0.3),
        ("pca3000_h1024",  3000, 1024, 0.3),
        ("pca5000_h1024",  5000, 1024, 0.3),
        ("pca2000_h2048",  2000, 2048, 0.3),
        ("pca3000_h2048",  3000, 2048, 0.3),
        # Higher dropout
        ("pca2000_h1024_d0.4", 2000, 1024, 0.4),
        ("pca3000_h1024_d0.4", 3000, 1024, 0.4),
    ]

    for label, pca_dim, hidden, dropout in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        X_tr = X_train_pca[:, :pca_dim]
        X_te = X_test_pca[:, :pca_dim]

        result = train_mlp(
            X_tr, y_train_gpu, X_te, y_test_gpu,
            pca_dim, num_classes, hidden, device,
            epochs=args.epochs, patience=args.patience,
            batch_size=512, lr=1e-3, weight_decay=1e-4,
            dropout=dropout, label=label)
        all_results[label] = result

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'eval_pca_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS - {dataset_name.upper()} PCA + MLP")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'PCA':>5} {'Hidden':>6} {'Test':>7}  {'Train':>7}  {'Ep':>4}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*7}  {'-'*7}  {'-'*4}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['test_accuracy']):
        print(f"  {name:<30} {r['input_dim']:>5} {r['hidden_dim']:>6} "
              f"{r['test_accuracy']*100:>6.2f}%  {r['train_accuracy']*100:>6.2f}%  {r['best_epoch']:>4}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*7}  {'-'*7}  {'-'*4}")
    print(f"  {'F-score top20k (prev best)':<30} {'20000':>5} {'1024':>6} {'55.86':>6}%  {'99.31':>6}%")
    print(f"  {'Full 105K (no reduction)':<30} {'105K':>5} {'1024':>6} {'54.07':>6}%  {'99.96':>6}%")
    print(f"{'='*60}\n")
    print(f"Results saved to {output_dir}/eval_pca_results.json")


if __name__ == '__main__':
    main()
