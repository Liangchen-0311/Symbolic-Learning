"""
VSR on Tabular Data: Physics-Based Regression.

Compares:
  1. MLP baseline (neural black-box)
  2. VSR: Polynomial + trig/exp features with sparse LASSO selection
  3. VSR-Net: Learned polynomial coefficients (task-based network from paper)

Datasets: 4 UCI physics datasets (Concrete, Airfoil, Energy, Power Plant)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import warnings
import os
import sys
import time
import urllib.request
import zipfile
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

DEVICE = 'cpu'
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==================== Data Download ====================

def download_datasets(data_dir='data'):
    """Download 4 UCI physics datasets."""
    os.makedirs(data_dir, exist_ok=True)
    datasets = {}

    # 1. Concrete Compressive Strength
    print("Downloading Concrete Strength...")
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        df = pd.read_excel(url)
        X = df.iloc[:, :-1].values.astype(np.float64)
        y = df.iloc[:, -1].values.astype(np.float64)
        feature_names = ['Cement', 'BlastFurnace', 'FlyAsh', 'Water',
                         'Superplast', 'CoarseAgg', 'FineAgg', 'Age']
        datasets['concrete'] = {'X': X, 'y': y, 'features': feature_names}
        print(f"  OK: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. Airfoil Self-Noise
    print("Downloading Airfoil Noise...")
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        df = pd.read_csv(url, sep='\t', header=None)
        X = df.iloc[:, :-1].values.astype(np.float64)
        y = df.iloc[:, -1].values.astype(np.float64)
        feature_names = ['Frequency', 'Angle', 'Chord', 'Velocity', 'Thickness']
        datasets['airfoil'] = {'X': X, 'y': y, 'features': feature_names}
        print(f"  OK: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. Energy Efficiency
    print("Downloading Energy Efficiency...")
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
        df = pd.read_excel(url)
        df = df.dropna()
        X = df.iloc[:, :-2].values.astype(np.float64)
        y = df.iloc[:, -2].values.astype(np.float64)  # Heating load
        feature_names = ['Compact', 'SurfArea', 'WallArea', 'RoofArea',
                         'Height', 'Orientation', 'GlazeArea', 'GlazeDist']
        datasets['energy'] = {'X': X, 'y': y, 'features': feature_names}
        print(f"  OK: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 4. Combined Cycle Power Plant
    print("Downloading Power Plant...")
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        resp = urllib.request.urlopen(url)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        # Find the xlsx file inside
        xlsx_name = [n for n in zf.namelist() if n.endswith('.xlsx')][0]
        df = pd.read_excel(io.BytesIO(zf.read(xlsx_name)))
        X = df.iloc[:, :-1].values.astype(np.float64)
        y = df.iloc[:, -1].values.astype(np.float64)
        feature_names = ['Temperature', 'ExhVacuum', 'AmbPressure', 'Humidity']
        datasets['power'] = {'X': X, 'y': y, 'features': feature_names}
        print(f"  OK: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  FAILED: {e}")

    print(f"\nDownloaded {len(datasets)}/4 datasets")
    return datasets


# ==================== VSR: Symbolic Feature Expansion ====================

def build_symbolic_features(X, feature_names, max_poly=3, include_trig=True, include_exp=True):
    """
    Build expanded symbolic feature matrix.

    This implements the VSR idea: test many candidate formulas
    by generating features like X^2, X^3, sin(X), exp(X), X_i * X_j,
    then let LASSO select the sparse best combination.

    Returns: (expanded_X, expanded_names)
    """
    n, d = X.shape
    features = []
    names = []

    # 1. Original features (degree 1)
    for i in range(d):
        features.append(X[:, i])
        names.append(feature_names[i])

    # 2. Polynomial features (degree 2, 3)
    for deg in range(2, max_poly + 1):
        for i in range(d):
            features.append(X[:, i] ** deg)
            names.append(f"{feature_names[i]}^{deg}")

    # 3. Pairwise interactions (X_i * X_j)
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X[:, i] * X[:, j])
            names.append(f"{feature_names[i]}*{feature_names[j]}")

    # 4. Pairwise ratios (X_i / X_j) — common in physics
    for i in range(d):
        for j in range(d):
            if i != j:
                denom = np.abs(X[:, j]) + 1e-8
                features.append(X[:, i] / denom)
                names.append(f"{feature_names[i]}/{feature_names[j]}")

    # 5. Trigonometric features
    if include_trig:
        for i in range(d):
            features.append(np.sin(X[:, i]))
            names.append(f"sin({feature_names[i]})")
            features.append(np.cos(X[:, i]))
            names.append(f"cos({feature_names[i]})")

    # 6. Exponential / log features
    if include_exp:
        for i in range(d):
            clipped = np.clip(X[:, i], -10, 10)
            features.append(np.exp(clipped))
            names.append(f"exp({feature_names[i]})")
            features.append(np.log(np.abs(X[:, i]) + 1e-8))
            names.append(f"log({feature_names[i]})")
            features.append(np.sqrt(np.abs(X[:, i])))
            names.append(f"sqrt({feature_names[i]})")

    expanded = np.column_stack(features)
    # Replace NaN/Inf
    expanded = np.nan_to_num(expanded, nan=0.0, posinf=1e6, neginf=-1e6)
    return expanded, names


def run_vsr(X_train, y_train, X_val, y_val, feature_names):
    """
    Run VSR: expand features symbolically, then use LASSO for sparse selection.
    Returns: (model, formula_str, selected_features)
    """
    # Build symbolic features
    X_train_exp, exp_names = build_symbolic_features(X_train, feature_names)
    X_val_exp, _ = build_symbolic_features(X_val, feature_names)

    # Normalize expanded features
    scaler = StandardScaler()
    X_train_exp = scaler.fit_transform(X_train_exp)
    X_val_exp = scaler.transform(X_val_exp)

    # LASSO with cross-validation for sparsity
    lasso = LassoCV(
        alphas=np.logspace(-5, 1, 50),
        cv=5,
        max_iter=10000,
        random_state=SEED
    )
    lasso.fit(X_train_exp, y_train)

    # Extract selected features (non-zero coefficients)
    coef = lasso.coef_
    nonzero = np.where(np.abs(coef) > 1e-6)[0]

    if len(nonzero) == 0:
        # Fallback to Ridge if LASSO kills everything
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
        ridge.fit(X_train_exp, y_train)
        coef = ridge.coef_
        nonzero = np.argsort(np.abs(coef))[-10:]  # top 10
        model = ridge
    else:
        model = lasso

    # Build formula string
    terms = []
    for idx in nonzero:
        c = coef[idx]
        terms.append(f"{c:+.3f}*{exp_names[idx]}")

    # Sort by absolute coefficient
    terms_sorted = sorted(
        [(np.abs(coef[idx]), f"{coef[idx]:+.3f}*{exp_names[idx]}") for idx in nonzero],
        reverse=True
    )
    formula_str = " ".join([t[1] for t in terms_sorted[:8]])  # top 8 terms
    if len(terms_sorted) > 8:
        formula_str += f" + {len(terms_sorted) - 8} more"

    return model, scaler, formula_str, len(nonzero)


# ==================== MLP Baseline ====================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, epochs=300, lr=0.001):
    """Train MLP with early stopping."""
    input_dim = X_train.shape[1]
    model = MLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_va = torch.FloatTensor(X_val).to(DEVICE)
    y_va = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 30
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = F.mse_loss(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = F.mse_loss(val_pred, y_va).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ==================== VSR-Net (Task-Based Network from Paper) ====================

class VSRNet(nn.Module):
    """
    Task-based network: applies polynomial transform then learns weights.
    Implements the VSR paper's approach: f(X) = (X^3 + X^2 + X) · w
    with learnable polynomial coefficients per feature.
    """
    def __init__(self, input_dim, max_order=3):
        super().__init__()
        self.input_dim = input_dim
        self.max_order = max_order
        # Learnable polynomial coefficients per feature per order
        self.poly_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) * 0.01)
            for _ in range(max_order + 1)
        ])
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Weighted polynomial: sum_k alpha_k * x^k for each feature
        transformed = torch.zeros_like(x)
        for k in range(self.max_order + 1):
            if k == 0:
                transformed = transformed + self.poly_weights[k].unsqueeze(0)
            else:
                transformed = transformed + self.poly_weights[k].unsqueeze(0) * (x ** k)
        return self.output(transformed).squeeze(-1)


def train_vsrnet(X_train, y_train, X_val, y_val, max_order=3, epochs=300, lr=0.005):
    """Train VSR-Net."""
    input_dim = X_train.shape[1]
    model = VSRNet(input_dim, max_order).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_va = torch.FloatTensor(X_val).to(DEVICE)
    y_va = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 30
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = F.mse_loss(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = F.mse_loss(val_pred, y_va).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ==================== Experiment Runner ====================

def run_one_dataset(name, X, y, feature_names):
    """Run full comparison on one dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    # Split: 70% train, 10% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED)

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    results = {}

    # --- 1. MLP Baseline ---
    print("\n[1/3] Training MLP baseline...")
    t0 = time.time()
    mlp, mlp_val = train_mlp(X_train_s, y_train_s, X_val_s, y_val_s)
    mlp_time = time.time() - t0

    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(torch.FloatTensor(X_test_s)).numpy()
    mlp_mse = mean_squared_error(y_test_s, mlp_pred)
    mlp_r2 = r2_score(y_test_s, mlp_pred)
    results['MLP'] = {'mse': mlp_mse, 'r2': mlp_r2, 'time': mlp_time}
    print(f"  MLP: MSE={mlp_mse:.6f}, R²={mlp_r2:.4f} ({mlp_time:.1f}s)")

    # --- 2. VSR (LASSO on expanded features) ---
    print("\n[2/3] Running VSR (symbolic feature expansion + LASSO)...")
    t0 = time.time()
    vsr_model, vsr_scaler, formula, n_terms = run_vsr(
        X_train_s, y_train_s, X_val_s, y_val_s, feature_names
    )
    vsr_time = time.time() - t0

    X_test_exp, _ = build_symbolic_features(X_test_s, feature_names)
    X_test_exp = vsr_scaler.transform(X_test_exp)
    vsr_pred = vsr_model.predict(X_test_exp)
    vsr_mse = mean_squared_error(y_test_s, vsr_pred)
    vsr_r2 = r2_score(y_test_s, vsr_pred)
    results['VSR'] = {'mse': vsr_mse, 'r2': vsr_r2, 'time': vsr_time, 'formula': formula, 'n_terms': n_terms}
    print(f"  VSR: MSE={vsr_mse:.6f}, R²={vsr_r2:.4f} ({n_terms} terms, {vsr_time:.1f}s)")
    print(f"  Formula: {formula}")

    # --- 3. VSR-Net (learned polynomial) ---
    print("\n[3/3] Training VSR-Net (task-based polynomial network)...")
    t0 = time.time()
    vsrnet, vsrnet_val = train_vsrnet(X_train_s, y_train_s, X_val_s, y_val_s)
    vsrnet_time = time.time() - t0

    vsrnet.eval()
    with torch.no_grad():
        vsrnet_pred = vsrnet(torch.FloatTensor(X_test_s)).numpy()
    vsrnet_mse = mean_squared_error(y_test_s, vsrnet_pred)
    vsrnet_r2 = r2_score(y_test_s, vsrnet_pred)
    results['VSR-Net'] = {'mse': vsrnet_mse, 'r2': vsrnet_r2, 'time': vsrnet_time}
    print(f"  VSR-Net: MSE={vsrnet_mse:.6f}, R²={vsrnet_r2:.4f} ({vsrnet_time:.1f}s)")

    # Summary for this dataset
    print(f"\n  {'Method':<12} {'MSE':<12} {'R²':<10} {'Time':<8}")
    print(f"  {'-'*42}")
    for method, r in results.items():
        print(f"  {method:<12} {r['mse']:<12.6f} {r['r2']:<10.4f} {r['time']:<8.1f}s")

    return results


def main():
    print("=" * 60)
    print("VSR on Tabular Data: Physics-Based Regression")
    print("=" * 60)

    # Download data
    print("\n--- DOWNLOADING DATASETS ---\n")
    datasets = download_datasets()

    if len(datasets) == 0:
        print("ERROR: No datasets downloaded. Check network connectivity.")
        return

    # Run experiments
    all_results = {}
    for name, data in datasets.items():
        result = run_one_dataset(name, data['X'], data['y'], data['features'])
        all_results[name] = result

    # ==================== Final Summary ====================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':<12} {'MLP MSE':<12} {'VSR MSE':<12} {'VSR-Net MSE':<14} {'Best':<10} {'VSR vs MLP':<12}")
    print("-" * 72)

    improvements = []
    for name, res in all_results.items():
        mlp_mse = res['MLP']['mse']
        vsr_mse = res['VSR']['mse']
        vsrnet_mse = res['VSR-Net']['mse']

        best = min(res.items(), key=lambda x: x[1]['mse'])
        imp = (mlp_mse - vsr_mse) / mlp_mse * 100
        improvements.append(imp)

        print(f"{name:<12} {mlp_mse:<12.6f} {vsr_mse:<12.6f} {vsrnet_mse:<14.6f} {best[0]:<10} {imp:+.1f}%")

    avg_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    print("-" * 72)
    print(f"{'Average VSR vs MLP improvement:':<52} {avg_imp:+.1f}% (±{std_imp:.1f}%)")

    # R² summary
    print(f"\n{'Dataset':<12} {'MLP R²':<10} {'VSR R²':<10} {'VSR-Net R²':<12}")
    print("-" * 44)
    for name, res in all_results.items():
        print(f"{name:<12} {res['MLP']['r2']:<10.4f} {res['VSR']['r2']:<10.4f} {res['VSR-Net']['r2']:<12.4f}")

    # Formulas
    print(f"\nDiscovered Formulas:")
    for name, res in all_results.items():
        if 'formula' in res['VSR']:
            print(f"  {name}: {res['VSR']['formula']}")

    # Verdict
    print(f"\n{'=' * 60}")
    if avg_imp > 15:
        print("VERDICT: VSR WINS (>15% average improvement)")
    elif avg_imp > 5:
        print("VERDICT: VSR SHOWS PROMISE (5-15% improvement)")
    elif avg_imp > 0:
        print("VERDICT: MIXED RESULTS (0-5% improvement)")
    else:
        print("VERDICT: MLP WINS (VSR did not improve)")
    print("=" * 60)


if __name__ == '__main__':
    main()
