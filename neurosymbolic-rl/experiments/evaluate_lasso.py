#!/usr/bin/env python3
"""
LASSO Feature Selection Evaluation.

Loads pre-extracted .npy feature matrices, runs L1-regularized Logistic
Regression (OneVsRest + liblinear) with multiple C values,
and reports Active Features count.

Usage:
    python experiments/evaluate_lasso.py --data_dir outputs/cifar100
"""

import os, sys, time, argparse, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description="LASSO feature selection eval")
    parser.add_argument('--data_dir', default='outputs/cifar100',
                        help='Directory containing X_train.npy etc.')
    args = parser.parse_args()

    d = args.data_dir
    print(f"Loading cached features from {d} ...", flush=True)
    X_train = np.load(os.path.join(d, 'X_train.npy'))
    y_train = np.load(os.path.join(d, 'y_train.npy'))
    X_test  = np.load(os.path.join(d, 'X_test.npy'))
    y_test  = np.load(os.path.join(d, 'y_test.npy'))
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}", flush=True)
    print(f"  Classes: {len(np.unique(y_train))}", flush=True)

    # StandardScaler
    print("\nStandardScaler ...", flush=True)
    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train))
    X_test_s  = np.nan_to_num(scaler.transform(X_test))

    # L1 sweep: C is inverse regularization strength (smaller C = stronger L1)
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    num_classes = len(np.unique(y_train))
    total_features = X_train_s.shape[1]

    print(f"\n{'='*76}", flush=True)
    print(f"  LASSO (L1) LogisticRegression — C-value sweep", flush=True)
    print(f"  OneVsRest + liblinear (penalty='l1'), max_iter=2000", flush=True)
    print(f"  {total_features} input features, {num_classes} classes", flush=True)
    print(f"{'='*76}", flush=True)
    print(f"  {'C':>6}  {'Train':>8}  {'Test':>8}  {'Gap':>8}  "
          f"{'Active':>12}  {'Pruned%':>8}  {'Time':>7}", flush=True)
    print(f"  {'-'*68}", flush=True)

    all_results = {}

    for C_val in C_values:
        t0 = time.time()
        base_clf = LogisticRegression(
            penalty='l1',
            C=C_val,
            solver='liblinear',
            max_iter=2000,
            tol=1e-4,
            random_state=42,
        )
        clf = OneVsRestClassifier(base_clf, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        fit_time = time.time() - t0

        tr_acc = clf.score(X_train_s, y_train)
        te_acc = clf.score(X_test_s, y_test)
        gap = tr_acc - te_acc

        # Active features: stack all per-class coef matrices
        # Each estimator has coef_ of shape (1, num_features)
        all_coefs = np.vstack([est.coef_ for est in clf.estimators_])  # (100, 2000)
        active_per_feature = np.any(all_coefs != 0, axis=0)
        n_active = int(active_per_feature.sum())
        pruned_pct = (1 - n_active / total_features) * 100

        print(f"  {C_val:>6.3f}  {tr_acc*100:>7.2f}%  {te_acc*100:>7.2f}%  "
              f"{gap*100:>7.2f}%  {n_active:>6}/{total_features}  "
              f"{pruned_pct:>7.1f}%  {fit_time:>6.1f}s", flush=True)

        all_results[f'L1_C={C_val}'] = {
            'C': C_val,
            'train_accuracy': float(tr_acc),
            'test_accuracy': float(te_acc),
            'gap': float(gap),
            'active_features': n_active,
            'total_features': total_features,
            'pruned_pct': float(pruned_pct),
            'fit_time_s': fit_time,
        }

    print(f"  {'-'*68}", flush=True)

    # Find best test accuracy
    best_key = max(all_results, key=lambda k: all_results[k]['test_accuracy'])
    best = all_results[best_key]
    print(f"  BEST: {best_key}  Test={best['test_accuracy']*100:.2f}%  "
          f"Active={best['active_features']}/{total_features}", flush=True)
    print(f"{'='*76}\n", flush=True)

    # Save
    results_path = os.path.join(d, 'eval_lasso_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}", flush=True)


if __name__ == '__main__':
    main()
