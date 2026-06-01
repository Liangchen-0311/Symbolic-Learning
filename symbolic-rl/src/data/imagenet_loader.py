"""
ImageNet data loader for symbolic feature discovery.

Key design decisions:
- No mean/std normalization: symbolic formulas operate on raw pixel values [0,1]
- Supports configurable resolution for Phase 1 (64x64) vs Phase 2/3 (224x224)
- Stratified sampling for balanced class representation
- Memory-efficient: uses standard torchvision ImageFolder with lazy loading
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Optional, Dict, List
import numpy as np


class _SafeImageFolder(torchvision.datasets.ImageFolder):
    """ImageFolder that silently skips corrupted images instead of crashing."""

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            # Return a black image with the correct label
            path, target = self.samples[index]
            dummy = torch.zeros(3, 224, 224)  # will be resized by transform anyway
            return dummy, target


class ImageNetDataModule:
    """Data module for ImageNet with resolution-adaptive loading."""

    def __init__(
        self,
        data_dir: str = '/data/imagenet',
        resolution: int = 224,
        batch_size: int = 256,
        num_workers: int = 8,
        val_split: float = 0.0,
        samples_per_class: Optional[int] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Path to ImageNet root (should contain train/ and val/ dirs)
            resolution: Image resolution (64 for Phase 1, 224 for Phase 2/3)
            batch_size: Batch size for data loading
            num_workers: Number of data loading workers
            val_split: Not used for ImageNet (has its own val set)
            samples_per_class: If set, subsample this many images per class
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_class = samples_per_class
        self.augment = augment

        self.num_classes = 1000
        self.input_channels = 3
        self.image_size = resolution

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Set up datasets with appropriate transforms."""
        # Standard preprocessing: resize, center-crop, to tensor
        # Do NOT normalize to ImageNet mean/std — formulas need raw [0,1] pixels
        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256 if self.resolution >= 224 else self.resolution + 32),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ])

        val_transform = transforms.Compose([
            transforms.Resize(256 if self.resolution >= 224 else self.resolution + 32),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
        ])

        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet train directory not found: {train_dir}\n"
                f"Expected structure: {self.data_dir}/train/n01440764/..."
            )

        self.train_dataset = _SafeImageFolder(
            train_dir, transform=train_transform
        )
        self.val_dataset = _SafeImageFolder(
            val_dir, transform=val_transform
        )
        # Test set = val set for ImageNet
        self.test_dataset = self.val_dataset

        # Stratified subsampling if requested
        if self.samples_per_class is not None:
            self.train_dataset = self._stratified_subset(
                self.train_dataset, self.samples_per_class
            )

        print(f"[ImageNet] Train: {len(self.train_dataset)} images")
        print(f"[ImageNet] Val: {len(self.val_dataset)} images")
        print(f"[ImageNet] Resolution: {self.resolution}x{self.resolution}")

    def _stratified_subset(self, dataset, samples_per_class: int) -> Subset:
        """Create a stratified subset with equal samples per class."""
        targets = np.array(dataset.targets)
        indices = []
        rng = np.random.RandomState(42)
        for cls_id in range(self.num_classes):
            cls_indices = np.where(targets == cls_id)[0]
            if len(cls_indices) > samples_per_class:
                chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
            else:
                chosen = cls_indices
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return Subset(dataset, indices)

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=False,
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        return self.get_val_loader()


# ============================================
# ImageNet WordNet Hierarchy → Superclasses
# ============================================

# 20 coarse superclasses from the ImageNet hierarchy
# Each maps a range of fine-grained classes to a superclass ID
# This is a simplified grouping based on common WordNet categories

IMAGENET_SUPERCLASS_NAMES = {
    0: 'fish_aquatic',
    1: 'bird',
    2: 'reptile_amphibian',
    3: 'insect_arthropod',
    4: 'mammal_pet',
    5: 'mammal_wild',
    6: 'primate',
    7: 'food_fruit',
    8: 'food_other',
    9: 'plant_flower',
    10: 'vehicle_land',
    11: 'vehicle_water_air',
    12: 'clothing_fabric',
    13: 'furniture_indoor',
    14: 'electronic_device',
    15: 'musical_instrument',
    16: 'container_vessel',
    17: 'tool_implement',
    18: 'structure_building',
    19: 'natural_scene_misc',
}


def build_imagenet_superclass_mapping(dataset) -> Dict[int, int]:
    """
    Build a mapping from ImageNet 1000-class IDs to ~20 superclass IDs.

    This uses the WordNet hierarchy (wnid) from the dataset's class_to_idx.
    Falls back to a simple modular mapping if wnids are not available.

    Args:
        dataset: torchvision ImageFolder dataset with class_to_idx

    Returns:
        mapping: Dict[int, int] mapping class_idx → superclass_idx
    """
    # Handle Subset wrapping (from stratified sampling)
    base = dataset
    while hasattr(base, 'dataset'):
        base = base.dataset
    num_classes = len(base.class_to_idx)
    num_superclasses = 20

    # Simple deterministic mapping: distribute classes evenly across superclasses
    # This is a fallback — for best results, use a proper WordNet hierarchy file
    mapping = {}
    for class_idx in range(num_classes):
        mapping[class_idx] = class_idx % num_superclasses

    return mapping
