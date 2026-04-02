"""
Tabular Data Module for physics and UCI datasets.

Supports 6 datasets:
- Concrete Compressive Strength (8 features, 1030 samples)
- Airfoil Self-Noise (5 features, 1503 samples)
- Energy Efficiency (8 features, 768 samples)
- Combined Cycle Power Plant (4 features, 9568 samples)
- Particle Collision (16 features, 10000 samples) - synthetic from M = sqrt(E1^2+E2^2-2E1E2cos(theta))
- Asteroid (19 features, 5000 samples) - synthetic from D = 1329 * 10^(-0.2H) / sqrt(albedo)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import List, Tuple
import urllib.request
import zipfile
import io
import os


DATASET_INFO = {
    'concrete': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
        'features': ['Cement', 'BlastFurnace', 'FlyAsh', 'Water',
                      'Superplast', 'CoarseAgg', 'FineAgg', 'Age'],
        'type': 'excel',
    },
    'airfoil': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
        'features': ['Frequency', 'Angle', 'Chord', 'Velocity', 'Thickness'],
        'type': 'tsv',
    },
    'energy': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
        'features': ['Compact', 'SurfArea', 'WallArea', 'RoofArea',
                      'Height', 'Orientation', 'GlazeArea', 'GlazeDist'],
        'type': 'excel_energy',
    },
    'power': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
        'features': ['Temperature', 'ExhVacuum', 'AmbPressure', 'Humidity'],
        'type': 'zip_excel',
    },
    'particle': {
        'features': ['E1', 'E2', 'Theta12', 'Px1', 'Py1', 'Pz1',
                      'Px2', 'Py2', 'Pz2', 'Eta1', 'Eta2', 'Phi1',
                      'Phi2', 'Charge1', 'Charge2', 'PT_ratio'],
        'type': 'synthetic_particle',
        'n_samples': 10000,
        'description': 'Particle Collision - invariant mass M=sqrt(E1^2+E2^2-2E1E2cos(theta))',
    },
    'asteroid': {
        'features': ['H', 'Albedo', 'SemiMajorAxis', 'Eccentricity',
                      'Inclination', 'AscNode', 'Perihelion', 'MeanAnomaly',
                      'Period', 'PerihelionDist', 'AphelionDist', 'MeanMotion',
                      'Condition', 'G_param', 'MOID', 'Jupiter_MOID',
                      'Epoch', 'TisserandJ', 'SpectralType'],
        'type': 'synthetic_asteroid',
        'n_samples': 5000,
        'description': 'Asteroid - diameter D=1329*10^(-0.2H)/sqrt(albedo)',
    },
}


class TabularDataModule:
    """
    Loads UCI tabular datasets with train/val/test splits.

    Interface matches what experiment scripts expect:
      - data_module.n_features
      - data_module.n_samples
      - data_module.train_X / train_y  (CPU tensors)
      - data_module.val_X / val_y
      - data_module.test_X / test_y
      - data_module.get_train_loader()
      - data_module.get_val_loader()
      - data_module.get_test_loader()
      - data_module.get_feature_names()
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 64,
        test_split: float = 0.2,
        val_split: float = 0.1,
        data_dir: str = 'data/tabular',
        random_seed: int = 42
    ):
        if dataset_name not in DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Available: {list(DATASET_INFO.keys())}")

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.random_seed = random_seed
        self._feature_names = DATASET_INFO[dataset_name]['features']
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load raw data
        X, y = self._load_dataset(dataset_name)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # Split 70/10/20
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed
        )
        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_seed
        )

        # Normalize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)

        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Convert to tensors (CPU)
        self.train_X = torch.FloatTensor(X_train)
        self.train_y = torch.FloatTensor(y_train)
        self.val_X = torch.FloatTensor(X_val)
        self.val_y = torch.FloatTensor(y_val)
        self.test_X = torch.FloatTensor(X_test)
        self.test_y = torch.FloatTensor(y_test)

        print(f"[TabularDataModule] {dataset_name}: "
              f"{self.n_samples} samples, {self.n_features} features")
        print(f"  Split: Train={len(self.train_X)}, Val={len(self.val_X)}, "
              f"Test={len(self.test_X)}")

    def _load_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Download and load dataset."""
        info = DATASET_INFO[name]
        cache_file = self.data_dir / f"{name}.npz"

        # Try loading from cache
        if cache_file.exists():
            data = np.load(cache_file)
            print(f"  Loaded {name} from cache")
            return data['X'], data['y']

        dtype = info['type']
        url = info.get('url', '')

        if dtype.startswith('synthetic_'):
            print(f"  Generating synthetic {name}...")
        else:
            print(f"  Downloading {name}...")

        if dtype == 'excel':
            df = pd.read_excel(url)
            X = df.iloc[:, :-1].values.astype(np.float64)
            y = df.iloc[:, -1].values.astype(np.float64)

        elif dtype == 'tsv':
            df = pd.read_csv(url, sep='\t', header=None)
            X = df.iloc[:, :-1].values.astype(np.float64)
            y = df.iloc[:, -1].values.astype(np.float64)

        elif dtype == 'excel_energy':
            df = pd.read_excel(url)
            df = df.dropna()
            X = df.iloc[:, :-2].values.astype(np.float64)
            y = df.iloc[:, -2].values.astype(np.float64)  # Heating load

        elif dtype == 'zip_excel':
            resp = urllib.request.urlopen(url)
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            xlsx_name = [n for n in zf.namelist() if n.endswith('.xlsx')][0]
            df = pd.read_excel(io.BytesIO(zf.read(xlsx_name)))
            X = df.iloc[:, :-1].values.astype(np.float64)
            y = df.iloc[:, -1].values.astype(np.float64)

        elif dtype == 'synthetic_particle':
            X, y = self._generate_particle_collision(info.get('n_samples', 10000))

        elif dtype == 'synthetic_asteroid':
            X, y = self._generate_asteroid(info.get('n_samples', 5000))

        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        # Cache
        np.savez(cache_file, X=X, y=y)
        print(f"  OK: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    @staticmethod
    def _generate_particle_collision(n_samples: int = 10000
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic particle collision data.

        True formula: M = sqrt(E1^2 + E2^2 - 2*E1*E2*cos(theta))
        16 features: E1, E2, theta12, momenta, pseudorapidity, etc.
        Most features are correlated noise; only E1, E2, theta matter.
        """
        rng = np.random.RandomState(42)

        # Core physics variables
        E1 = rng.uniform(1, 100, n_samples)
        E2 = rng.uniform(1, 100, n_samples)
        theta12 = rng.uniform(0.1, np.pi - 0.1, n_samples)

        # True invariant mass
        mass = np.sqrt(E1**2 + E2**2 - 2 * E1 * E2 * np.cos(theta12))

        # Derived / correlated features (realistic but not needed for formula)
        P1 = np.sqrt(np.maximum(E1**2 - 0.105**2, 0))  # ~massless approx
        P2 = np.sqrt(np.maximum(E2**2 - 0.105**2, 0))
        phi1 = rng.uniform(0, 2 * np.pi, n_samples)
        phi2 = phi1 + theta12 + rng.normal(0, 0.05, n_samples)
        eta1 = rng.uniform(-2.5, 2.5, n_samples)
        eta2 = rng.uniform(-2.5, 2.5, n_samples)

        Px1 = P1 * np.sin(np.arctan(np.exp(-eta1)) * 2) * np.cos(phi1)
        Py1 = P1 * np.sin(np.arctan(np.exp(-eta1)) * 2) * np.sin(phi1)
        Pz1 = P1 * np.cos(np.arctan(np.exp(-eta1)) * 2)
        Px2 = P2 * np.sin(np.arctan(np.exp(-eta2)) * 2) * np.cos(phi2)
        Py2 = P2 * np.sin(np.arctan(np.exp(-eta2)) * 2) * np.sin(phi2)
        Pz2 = P2 * np.cos(np.arctan(np.exp(-eta2)) * 2)

        charge1 = rng.choice([-1, 1], n_samples).astype(float)
        charge2 = -charge1  # opposite charge
        pt_ratio = (P1 * np.sin(np.arctan(np.exp(-eta1)) * 2)) / \
                   (P2 * np.sin(np.arctan(np.exp(-eta2)) * 2) + 1e-8)

        # 16 features
        X = np.column_stack([
            E1, E2, theta12,
            Px1, Py1, Pz1, Px2, Py2, Pz2,
            eta1, eta2, phi1, phi2,
            charge1, charge2, pt_ratio
        ])

        # Add small noise to target for realism
        mass += rng.normal(0, 0.5, n_samples)
        mass = np.maximum(mass, 0.1)

        print(f"  Generated particle collision: {n_samples} samples, "
              f"16 features, target=invariant mass")
        return X, mass

    @staticmethod
    def _generate_asteroid(n_samples: int = 5000
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic asteroid data.

        True formula: D = 1329 * 10^(-0.2*H) / sqrt(albedo)
        19 features: H, albedo, orbital params, etc.
        Only H and albedo matter for the true formula.
        """
        rng = np.random.RandomState(42)

        # Core physics variables
        H = rng.uniform(10, 25, n_samples)          # absolute magnitude
        albedo = rng.uniform(0.03, 0.5, n_samples)  # geometric albedo

        # True diameter formula
        diameter = 1329.0 * np.power(10, -0.2 * H) / np.sqrt(albedo)

        # Correlated orbital features (realistic ranges)
        semi_major = rng.uniform(0.5, 5.2, n_samples)       # AU
        eccentricity = rng.beta(2, 5, n_samples)             # 0-1
        inclination = rng.exponential(10, n_samples)          # degrees
        asc_node = rng.uniform(0, 360, n_samples)
        perihelion_arg = rng.uniform(0, 360, n_samples)
        mean_anomaly = rng.uniform(0, 360, n_samples)
        period = semi_major**1.5  # Kepler's third law (years)
        perihelion_dist = semi_major * (1 - eccentricity)
        aphelion_dist = semi_major * (1 + eccentricity)
        mean_motion = 360.0 / (period * 365.25)  # deg/day
        condition = rng.uniform(0, 9, n_samples)  # orbit condition code
        G_param = rng.normal(0.15, 0.1, n_samples)  # slope parameter
        MOID = rng.exponential(0.3, n_samples)  # Earth MOID (AU)
        jupiter_MOID = rng.exponential(1.0, n_samples)
        epoch = rng.uniform(58000, 60000, n_samples)  # MJD
        tisserand_J = 3.0 - semi_major / 5.2 - \
            2 * np.sqrt(semi_major / 5.2 * (1 - eccentricity**2)) * \
            np.cos(np.radians(inclination))
        spectral_type = rng.choice([0, 1, 2, 3], n_samples).astype(float)  # S/C/M/X encoded

        # 19 features
        X = np.column_stack([
            H, albedo, semi_major, eccentricity, inclination,
            asc_node, perihelion_arg, mean_anomaly, period,
            perihelion_dist, aphelion_dist, mean_motion,
            condition, G_param, MOID, jupiter_MOID,
            epoch, tisserand_J, spectral_type
        ])

        # Add noise to target
        diameter += rng.normal(0, diameter * 0.05)  # 5% noise
        diameter = np.maximum(diameter, 0.001)

        print(f"  Generated asteroid: {n_samples} samples, "
              f"19 features, target=diameter")
        return X, diameter

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def get_train_loader(self) -> DataLoader:
        ds = TensorDataset(self.train_X, self.train_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> DataLoader:
        ds = TensorDataset(self.val_X, self.val_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self) -> DataLoader:
        ds = TensorDataset(self.test_X, self.test_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)
