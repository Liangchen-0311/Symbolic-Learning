"""
Data module for image classification datasets.

Supports: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split


class MNISTDataModule:
    """Unified data module for image classification benchmarks."""

    DATASET_MAP = {
        'mnist':         (torchvision.datasets.MNIST, 1, 28, 10),
        'fashion_mnist': (torchvision.datasets.FashionMNIST, 1, 28, 10),
        'cifar10':       (torchvision.datasets.CIFAR10, 3, 32, 10),
        'cifar100':      (torchvision.datasets.CIFAR100, 3, 32, 100),
    }

    def __init__(
        self,
        dataset='cifar10',
        batch_size=128,
        num_workers=0,
        data_dir='./data',
        val_split=0.1,
        train_subset=None,
        test_subset=None,
        augment=False,
        # Legacy aliases accepted but ignored
        **kwargs,
    ):
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.val_split = val_split
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.augment = augment

        ds_cls, channels, size, n_classes = self.DATASET_MAP[dataset]
        self._ds_cls = ds_cls
        self.input_channels = channels
        self.image_size = size
        self.num_classes = n_classes

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        test_transform = transforms.Compose([transforms.ToTensor()])

        if self.augment and self.image_size == 32:
            # Data augmentation for CIFAR (32x32 images)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            print(f"  [Data] Augmentation enabled: RandomCrop(32,pad=4) + HorizontalFlip")
        else:
            train_transform = test_transform

        full_train = self._ds_cls(
            root=self.data_dir, train=True, download=True, transform=train_transform,
        )
        self.test_dataset = self._ds_cls(
            root=self.data_dir, train=False, download=True, transform=test_transform,
        )

        # Optional subset
        if self.train_subset is not None:
            n = min(int(self.train_subset), len(full_train))
            full_train = Subset(full_train, list(range(n)))

        if self.test_subset is not None:
            n = min(int(self.test_subset), len(self.test_dataset))
            self.test_dataset = Subset(self.test_dataset, list(range(n)))

        # Train / val split
        n_val = int(len(full_train) * self.val_split)
        n_train = len(full_train) - n_val
        self.train_dataset, self.val_dataset = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # ---- DataLoaders ------------------------------------------------
    def get_train_loader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )
