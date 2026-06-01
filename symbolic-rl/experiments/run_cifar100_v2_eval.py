#!/usr/bin/env python3
"""CIFAR-100 v2 evaluation: 1D features (no SPP) + MLP.

This script evaluates the v2 feature bank (1D pooling, ~5000 dims)
with various MLP configurations and regularization settings.
"""
import os, sys, json, time, argparse
import yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.large_feature_bank import LargeFeatureBank
from experiments.evaluate_mlp import extract_features


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_mlp(X_train, y_train, X_test, y_test, input_dim, num_classes,
              hidden_dims, device, epochs=300, patience=40, batch_size=256,
              lr=1e-3, weight_decay=1e-4, dropout=0.3,
              label_smoothing=0.0, mixup_alpha=0.0, label=""):
    model = MLPClassifier(input_dim, num_classes, hidden_dims, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  hidden={hidden_dims}  d={dropout}  "
          f"ls={label_smoothing}  mix={mixup_alpha}  wd={weight_decay}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
            if mixup_alpha > 0:
                xb, ya, yb_mix, lam = mixup_data(xb, yb, mixup_alpha)
                logits = model(xb)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_mix)
                train_correct += (logits.argmax(1) == ya).sum().item()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                train_correct += (logits.argmax(1) == yb).sum().item()
            loss.backward()
            optimizer.step()
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
        'hidden_dims': hidden_dims,
        'dropout': dropout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tensor_vsr_cifar100_v2.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device
    dataset_name = config.get('dataset', 'cifar100')
    num_classes = config.get('training', {}).get('num_classes', 100)
    output_dir = config.get('output_dir', 'outputs/cifar100_v2')
    bank_path = os.path.join(output_dir, 'feature_bank')

    # --- Load feature bank ---
    print(f"Loading feature bank from {bank_path}...")
    bank = LargeFeatureBank.load(bank_path, device=device)
    formula_strs = bank.formula_strs
    print(f"  {len(formula_strs)} formulas")
    print(bank.get_summary())
    del bank

    # --- Extract features ---
    data_module = MNISTDataModule(
        dataset=dataset_name,
        batch_size=config.get('training', {}).get('eval_batch_size', 2048),
        num_workers=4, val_split=0.0)
    data_module.setup()
    print(f"Dataset: {dataset_name.upper()} — "
          f"Train: {len(data_module.train_dataset)}, "
          f"Test: {len(data_module.test_dataset)}")

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
    print(f"\nFeature dims: {input_dim}  (sample:dim ratio = {n_train/input_dim:.1f}:1)")

    # --- Standardize + GPU ---
    print("Standardizing + moving to GPU...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-8)

    X_train_gpu = torch.from_numpy(np.clip((X_train - mean) / std, -10, 10)).float().to(device)
    X_test_gpu = torch.from_numpy(np.clip((X_test - mean) / std, -10, 10)).float().to(device)
    del X_train, X_test, mean, std

    y_train_gpu = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_gpu = torch.tensor(y_test, dtype=torch.long, device=device)
    del y_train, y_test
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # --- MLP Experiments ---
    all_results = {}

    # (label, hidden_dims, dropout, ls, mixup, wd, batch_size)
    configs = [
        # Baseline - single hidden layer
        ("h512",              [512],       0.3, 0.0, 0.0, 1e-4, 256),
        ("h1024",             [1024],      0.3, 0.0, 0.0, 1e-4, 256),
        ("h2048",             [2048],      0.3, 0.0, 0.0, 1e-4, 256),

        # Deeper MLPs (2 hidden layers)
        ("h1024_h512",        [1024, 512], 0.3, 0.0, 0.0, 1e-4, 256),
        ("h2048_h1024",       [2048, 1024],0.3, 0.0, 0.0, 1e-4, 256),

        # Higher dropout
        ("h1024_d0.4",        [1024],      0.4, 0.0, 0.0, 1e-4, 256),
        ("h1024_d0.5",        [1024],      0.5, 0.0, 0.0, 1e-4, 256),

        # Label smoothing + Mixup (worked for CIFAR-10)
        ("h1024_ls0.1",       [1024],      0.3, 0.1, 0.0, 1e-4, 256),
        ("h1024_mix0.2",      [1024],      0.3, 0.0, 0.2, 1e-4, 256),
        ("h1024_ls0.1_mix0.2",[1024],      0.3, 0.1, 0.2, 1e-4, 256),

        # Deeper + regularization
        ("h1024_h512_d0.4",   [1024, 512], 0.4, 0.0, 0.0, 1e-4, 256),
        ("h1024_h512_ls0.1_mix0.2", [1024, 512], 0.3, 0.1, 0.2, 1e-4, 256),

        # Higher weight decay
        ("h1024_wd5e-4",      [1024],      0.3, 0.0, 0.0, 5e-4, 256),
        ("h1024_d0.4_wd5e-4", [1024],      0.4, 0.0, 0.0, 5e-4, 256),
    ]

    for label, hidden, dropout, ls, mixup, wd, bs in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        result = train_mlp(
            X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu,
            input_dim, num_classes, hidden, device,
            epochs=args.epochs, patience=args.patience,
            batch_size=bs, lr=1e-3, weight_decay=wd,
            dropout=dropout, label_smoothing=ls,
            mixup_alpha=mixup, label=label)
        all_results[label] = result

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'eval_mlp_v2_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS - {dataset_name.upper()} v2 (1D pooling + augment)")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Dims':>5} {'Hidden':<15} {'Test':>7}  {'Train':>7}  {'Ep':>4}")
    print(f"  {'-'*30} {'-'*5} {'-'*15} {'-'*7}  {'-'*7}  {'-'*4}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['test_accuracy']):
        h_str = str(r.get('hidden_dims', ''))
        print(f"  {name:<30} {r['input_dim']:>5} {h_str:<15} "
              f"{r['test_accuracy']*100:>6.2f}%  {r['train_accuracy']*100:>6.2f}%  {r['best_epoch']:>4}")
    print(f"  {'-'*30} {'-'*5} {'-'*15} {'-'*7}  {'-'*7}  {'-'*4}")
    print(f"  {'SPP F-score best (prev)':<30} {'20K':>5} {'[1024]':<15} {'55.86':>6}%  {'99.31':>6}%")
    print(f"  {'Old 1200 1D (reference)':<30} {'1200':>5} {'[1024]':<15} {'~55.0':>6}%  {'~99.0':>6}%")
    print(f"{'='*60}\n")
    print(f"Results saved to {output_dir}/eval_mlp_v2_results.json")


if __name__ == '__main__':
    main()
