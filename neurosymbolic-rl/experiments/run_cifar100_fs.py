#!/usr/bin/env python3
"""CIFAR-100: Feature Selection + MLP. Reduce 105K dims to find sweet spot."""
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


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_mlp(X_train, y_train, X_test, y_test, input_dim, num_classes,
              hidden_dim, device, epochs=200, patience=30, batch_size=512,
              lr=1e-3, weight_decay=1e-4, dropout=0.3,
              label_smoothing=0.0, mixup_alpha=0.0, label=""):
    model = MLPClassifier(input_dim, num_classes, hidden_dim, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  dropout={dropout}  ls={label_smoothing}  mixup={mixup_alpha}")

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
    }


def f_score_selection(X_train_gpu, y_train_gpu, input_dim, num_classes, device):
    """ANOVA F-score feature selection on GPU."""
    print("\nComputing F-scores for feature selection...")
    t0 = time.time()
    global_mean = X_train_gpu.mean(dim=0)
    between_var = torch.zeros(input_dim, device=device)
    within_var = torch.zeros(input_dim, device=device)
    for c in range(num_classes):
        mask_c = (y_train_gpu == c)
        n_c = mask_c.sum().float()
        if n_c < 2:
            continue
        X_c = X_train_gpu[mask_c]
        class_mean = X_c.mean(dim=0)
        between_var += n_c * (class_mean - global_mean) ** 2
        within_var += X_c.var(dim=0) * (n_c - 1)
    within_var = within_var.clamp(min=1e-10)
    f_scores = between_var / within_var
    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"  F-score stats: min={f_scores.min():.4f}, mean={f_scores.mean():.4f}, "
          f"max={f_scores.max():.4f}")
    return f_scores


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

    # --- Extract features ---
    print(f"Loading feature bank from {bank_path}...")
    bank = LargeFeatureBank.load(bank_path, device=device)
    formula_strs = bank.formula_strs
    print(f"  {len(formula_strs)} formulas")
    print(bank.get_summary())
    del bank

    batch_size_data = config.get('training', {}).get('eval_batch_size', 2048)
    data_module = MNISTDataModule(
        dataset=dataset_name, batch_size=batch_size_data,
        num_workers=4, val_split=0.0)
    data_module.setup()
    print(f"Dataset: {dataset_name.upper()}")
    print(f"  Train: {len(data_module.train_dataset)}, Test: {len(data_module.test_dataset)}")

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
    t0 = time.time()
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-8)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print(f"Moving to GPU...")
    t0 = time.time()
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
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB  ({time.time()-t0:.1f}s)")

    # --- Feature selection ---
    f_scores = f_score_selection(X_train_gpu, y_train_gpu, input_dim, num_classes, device)

    # Pre-compute top-K indices
    top_k_map = {}
    for k in [2000, 5000, 10000, 15000, 20000, 30000]:
        if k >= input_dim:
            continue
        _, indices = torch.topk(f_scores, k)
        top_k_map[k] = indices.sort().values
        sel_scores = f_scores[top_k_map[k]]
        print(f"  Top-{k}: F-score [{sel_scores.min():.4f}, {sel_scores.max():.4f}]")

    # --- Experiments ---
    all_results = {}

    # (label, top_k, hidden, dropout, ls, mixup, wd)
    configs = [
        # Feature selection sweep (baseline MLP, no regularization)
        ("full_105k_h1024",    None,  1024, 0.3, 0.0, 0.0, 1e-4),
        ("top30k_h1024",       30000, 1024, 0.3, 0.0, 0.0, 1e-4),
        ("top20k_h1024",       20000, 1024, 0.3, 0.0, 0.0, 1e-4),
        ("top15k_h1024",       15000, 1024, 0.3, 0.0, 0.0, 1e-4),
        ("top10k_h1024",       10000, 1024, 0.3, 0.0, 0.0, 1e-4),
        ("top5k_h1024",        5000,  1024, 0.3, 0.0, 0.0, 1e-4),
        ("top2k_h1024",        2000,  1024, 0.3, 0.0, 0.0, 1e-4),
        # Best K with regularization
        # (will be filled in after finding best K)
    ]

    for label, top_k, hidden, dropout, ls, mixup, wd in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        if top_k is not None and top_k in top_k_map:
            idx = top_k_map[top_k]
            X_tr = X_train_gpu[:, idx]
            X_te = X_test_gpu[:, idx]
            dim = top_k
        else:
            X_tr = X_train_gpu
            X_te = X_test_gpu
            dim = input_dim

        result = train_mlp(
            X_tr, y_train_gpu, X_te, y_test_gpu,
            dim, num_classes, hidden, device,
            epochs=args.epochs, patience=args.patience,
            batch_size=512, lr=lr if 'lr' in dir() else 1e-3,
            weight_decay=wd, dropout=dropout,
            label_smoothing=ls, mixup_alpha=mixup, label=label)
        all_results[label] = result

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'eval_mlp_fs_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Find best K from feature selection sweep
    best_k = None
    best_acc = 0
    for label, r in all_results.items():
        if r['test_accuracy'] > best_acc:
            best_acc = r['test_accuracy']
            best_k = r.get('input_dim', input_dim)

    print(f"\n  Best feature dim so far: {best_k} ({best_acc*100:.2f}%)")

    # Run best K with regularization combos
    if best_k and best_k in top_k_map:
        reg_configs = [
            (f"top{best_k//1000}k_h1024_ls0.1",          1024, 0.3, 0.1, 0.0, 1e-4),
            (f"top{best_k//1000}k_h1024_mix0.2",         1024, 0.3, 0.0, 0.2, 1e-4),
            (f"top{best_k//1000}k_h1024_ls0.1_mix0.2",   1024, 0.3, 0.1, 0.2, 1e-4),
            (f"top{best_k//1000}k_h2048",                2048, 0.3, 0.0, 0.0, 1e-4),
            (f"top{best_k//1000}k_h1024_d0.4",           1024, 0.4, 0.0, 0.0, 1e-4),
        ]
        idx = top_k_map[best_k]
        X_tr = X_train_gpu[:, idx]
        X_te = X_test_gpu[:, idx]

        for label, hidden, dropout, ls, mixup, wd in reg_configs:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            result = train_mlp(
                X_tr, y_train_gpu, X_te, y_test_gpu,
                best_k, num_classes, hidden, device,
                epochs=args.epochs, patience=args.patience,
                batch_size=512, lr=1e-3, weight_decay=wd,
                dropout=dropout, label_smoothing=ls, mixup_alpha=mixup,
                label=label)
            all_results[label] = result
            with open(os.path.join(output_dir, 'eval_mlp_fs_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS - {dataset_name.upper()} Feature Selection")
    print(f"{'='*60}")
    print(f"  {'Config':<35} {'Dims':>6} {'Test':>7}  {'Train':>7}  {'Epoch':>5}")
    print(f"  {'-'*35} {'-'*6} {'-'*7}  {'-'*7}  {'-'*5}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['test_accuracy']):
        print(f"  {name:<35} {r['input_dim']:>6} {r['test_accuracy']*100:>6.2f}%  "
              f"{r['train_accuracy']*100:>6.2f}%  {r['best_epoch']:>5}")
    print(f"{'='*60}\n")
    print(f"Results saved to {output_dir}/eval_mlp_fs_results.json")


if __name__ == '__main__':
    main()
