#!/usr/bin/env python3
"""CIFAR-100: Combined feature banks evaluation.

Combines formulas from multiple banks and evaluates with MLP.
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.data.mnist_loader import MNISTDataModule
from experiments.evaluate_mlp import extract_features


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
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
    print(f"  Params: {n_params:,}  h={hidden_dims}  d={dropout}  ls={label_smoothing}  mix={mixup_alpha}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    n_train = X_train.shape[0]
    best_test_acc, best_epoch, no_improve = 0.0, 0, 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_correct = train_total = 0
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
            test_correct = test_total = 0
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
    }


def f_score_selection(X, y, num_classes, device):
    """GPU F-score feature selection."""
    global_mean = X.mean(dim=0)
    between = torch.zeros(X.shape[1], device=device)
    within = torch.zeros(X.shape[1], device=device)
    for c in range(num_classes):
        mask = (y == c)
        nc = mask.sum().float()
        if nc < 2:
            continue
        Xc = X[mask]
        cm = Xc.mean(dim=0)
        between += nc * (cm - global_mean) ** 2
        within += Xc.var(dim=0) * (nc - 1)
    return between / within.clamp(min=1e-10)


def load_formulas(bank_path):
    """Load formula strings from a feature bank."""
    with open(os.path.join(bank_path, 'feature_bank.json')) as f:
        bank = json.load(f)
    return [f['str'] for f in bank['formulas']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    args = parser.parse_args()

    device = args.device
    num_classes = 100
    output_dir = 'outputs/cifar100_combo'
    os.makedirs(output_dir, exist_ok=True)

    # --- Load formulas from all banks ---
    bank_configs = [
        ('v2', 'outputs/cifar100_v2/feature_bank'),
        ('old', 'outputs/cifar100/feature_bank'),
        ('spp', 'outputs/cifar100_spp/feature_bank'),
    ]

    all_bank_formulas = {}
    for name, path in bank_configs:
        formulas = load_formulas(path)
        all_bank_formulas[name] = formulas
        spp_count = sum(1 for f in formulas if 'spp_pool' in f)
        dims = spp_count * 21 + (len(formulas) - spp_count)
        print(f"  [{name}] {len(formulas)} formulas, {dims} dims")

    # --- Define combinations ---
    combos = {
        'v2_only':       ['v2'],
        'old_only':      ['old'],
        'v2+old':        ['v2', 'old'],
        'v2+old+spp':    ['v2', 'old', 'spp'],
    }

    # --- Load data ---
    dm = MNISTDataModule(dataset='cifar100', batch_size=2048, num_workers=4, val_split=0.0)
    dm.setup()
    print(f"Dataset: CIFAR-100 — Train: {len(dm.train_dataset)}, Test: {len(dm.test_dataset)}")

    all_results = {}

    for combo_name, bank_names in combos.items():
        print(f"\n{'='*70}")
        print(f"  Combination: {combo_name}")
        print(f"{'='*70}")

        # Merge formulas (deduplicate)
        seen = set()
        merged = []
        for bn in bank_names:
            for f in all_bank_formulas[bn]:
                if f not in seen:
                    seen.add(f)
                    merged.append(f)

        spp_n = sum(1 for f in merged if 'spp_pool' in f)
        total_dims = spp_n * 21 + (len(merged) - spp_n)
        print(f"  Merged: {len(merged)} unique formulas, {total_dims} dims "
              f"(1D: {len(merged)-spp_n}, SPP: {spp_n})")

        # Extract features
        print("  Extracting train features...")
        t0 = time.time()
        X_train, y_train = extract_features(merged, dm.get_train_loader(), device, tag="train")
        print(f"    {X_train.shape} ({time.time()-t0:.1f}s)")

        print("  Extracting test features...")
        t0 = time.time()
        X_test, y_test = extract_features(merged, dm.get_test_loader(), device, tag="test")
        print(f"    {X_test.shape} ({time.time()-t0:.1f}s)")

        input_dim = X_train.shape[1]

        # Standardize + GPU
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        mean = X_train.mean(axis=0)
        std = np.maximum(X_train.std(axis=0), 1e-8)

        # Move to GPU in chunks if large
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        chunk = 5000
        X_tr_gpu = torch.empty(n_train, input_dim, dtype=torch.float32, device=device)
        for s in range(0, n_train, chunk):
            e = min(s + chunk, n_train)
            block = np.clip((X_train[s:e] - mean) / std, -10, 10)
            X_tr_gpu[s:e] = torch.from_numpy(block).to(device)
        del X_train

        X_te_gpu = torch.empty(n_test, input_dim, dtype=torch.float32, device=device)
        for s in range(0, n_test, chunk):
            e = min(s + chunk, n_test)
            block = np.clip((X_test[s:e] - mean) / std, -10, 10)
            X_te_gpu[s:e] = torch.from_numpy(block).to(device)
        del X_test, mean, std

        y_tr_gpu = torch.tensor(y_train, dtype=torch.long, device=device)
        y_te_gpu = torch.tensor(y_test, dtype=torch.long, device=device)
        del y_train, y_test
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # F-score selection if dims > 5000
        if input_dim > 5000:
            print(f"  Computing F-scores for feature selection...")
            f_scores = f_score_selection(X_tr_gpu, y_tr_gpu, num_classes, device)

            # Test various K
            for k in [3000, 5000, 10000, 20000]:
                if k >= input_dim:
                    continue
                _, topk_idx = torch.topk(f_scores, k)
                topk_idx = topk_idx.sort().values

                label = f"{combo_name}_top{k//1000}k"
                print(f"\n  --- {label} ---")

                result = train_mlp(
                    X_tr_gpu[:, topk_idx], y_tr_gpu,
                    X_te_gpu[:, topk_idx], y_te_gpu,
                    k, num_classes, [1024, 512], device,
                    epochs=args.epochs, patience=args.patience,
                    dropout=0.4, label=label)
                all_results[label] = result

                with open(os.path.join(output_dir, 'combo_results.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)

        # Full dims (only if manageable)
        if input_dim <= 10000:
            label = f"{combo_name}_full"
            print(f"\n  --- {label} ({input_dim} dims) ---")

            # Try best configs
            for suffix, hidden, dropout, ls, mix in [
                ("_h1024_h512_d0.4", [1024, 512], 0.4, 0.0, 0.0),
                ("_h1024_d0.5",      [1024],      0.5, 0.0, 0.0),
                ("_h2048",           [2048],       0.3, 0.0, 0.0),
            ]:
                lbl = f"{combo_name}{suffix}"
                print(f"\n  --- {lbl} ---")
                result = train_mlp(
                    X_tr_gpu, y_tr_gpu, X_te_gpu, y_te_gpu,
                    input_dim, num_classes, hidden, device,
                    epochs=args.epochs, patience=args.patience,
                    dropout=dropout, label_smoothing=ls,
                    mixup_alpha=mix, label=lbl)
                all_results[lbl] = result

                with open(os.path.join(output_dir, 'combo_results.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)

        # Free GPU
        del X_tr_gpu, X_te_gpu, y_tr_gpu, y_te_gpu
        torch.cuda.empty_cache()

    # Final report
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — CIFAR-100 Combined Banks")
    print(f"{'='*70}")
    print(f"  {'Config':<35} {'Dims':>6} {'Test':>7}  {'Train':>7}  {'Ep':>4}")
    print(f"  {'-'*35} {'-'*6} {'-'*7}  {'-'*7}  {'-'*4}")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['test_accuracy']):
        print(f"  {name:<35} {r['input_dim']:>6} {r['test_accuracy']*100:>6.2f}%  "
              f"{r['train_accuracy']*100:>6.2f}%  {r['best_epoch']:>4}")
    print(f"  {'-'*35} {'-'*6} {'-'*7}  {'-'*7}  {'-'*4}")
    print(f"  {'v2 best (prev)':<35} {'1726':>6} {'55.00':>6}%  {'87.83':>6}%")
    print(f"  {'SPP F-score best (prev)':<35} {'20K':>6} {'55.86':>6}%  {'99.31':>6}%")
    print(f"{'='*70}\n")
    print(f"Results saved to {output_dir}/combo_results.json")


if __name__ == '__main__':
    main()
