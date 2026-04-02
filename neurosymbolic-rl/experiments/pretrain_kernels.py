#!/usr/bin/env python3
"""
Pretrain learnable kernels via supervised classification.

Model: For each of 10 terminal channels, apply 18 kernels → adaptive_avg_pool → Linear(180, 1000)
Data: 20K images (20/class), 112×112
Output: kernel_bank_pretrained.pt

This replaces random kernel initialization with meaningful filters that
capture useful visual patterns (edges, textures, etc.) before RL starts.
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.imagenet_loader import ImageNetDataModule
from src.symbolic.tensor_operators import SymbolicKernelBank

DEVICE = 'cuda'
DATA_DIR = '/workspace/neurosymbolic-rl/data/imagenet'
OUTPUT_PATH = 'outputs/imagenet_v3/kernel_bank_pretrained.pt'


class KernelPretrainer(nn.Module):
    """Wraps SymbolicKernelBank in a classification model for pretraining.

    Only pretrains the 12 LEARNABLE kernels (conv3x3_0..7, conv5x5_0..3).
    Classic kernels (edge_x, edge_y, laplacian, gabor_0/45/90) are already
    meaningful fixed filters — they don't need pretraining.

    Model: 10 terminals × 12 learnable kernels → pool → Linear(120, 1000)
    """

    def __init__(self, kernel_bank, n_terminals=10, num_classes=1000):
        super().__init__()
        self.kb = kernel_bank
        self.n_terminals = n_terminals
        # Only learnable kernels: 8 conv3x3 + 4 conv5x5 = 12
        self.n_learnable = kernel_bank.conv3x3.shape[0] + kernel_bank.conv5x5.shape[0]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.n_learnable * n_terminals, num_classes)

        # Freeze classic kernels — they are already Sobel/Gabor values
        kernel_bank.classic_3x3.requires_grad_(False)
        kernel_bank.classic_7x7.requires_grad_(False)

    def apply_learnable_kernels(self, x):
        """Apply 12 learnable kernels to single-channel input [B,H,W] → [B, 12]."""
        x4d = x.unsqueeze(1)  # [B, 1, H, W]
        features = []

        # Conv3x3 (8 learnable kernels)
        for i in range(self.kb.conv3x3.shape[0]):
            kernel = self.kb.conv3x3[i:i+1]
            out = F.conv2d(x4d, kernel, padding=1)
            features.append(self.pool(out).flatten(1))

        # Conv5x5 (4 learnable kernels)
        for i in range(self.kb.conv5x5.shape[0]):
            kernel = self.kb.conv5x5[i:i+1]
            out = F.conv2d(x4d, kernel, padding=2)
            features.append(self.pool(out).flatten(1))

        return torch.cat(features, dim=1)  # [B, 12]

    def forward(self, terminal_dict):
        """terminal_dict: {name: [B, H, W]} → logits [B, 1000]."""
        all_feats = []
        for name in sorted(terminal_dict.keys()):
            channel = terminal_dict[name]
            feats = self.apply_learnable_kernels(channel)  # [B, 12]
            all_feats.append(feats)
        return self.fc(torch.cat(all_feats, dim=1))  # [B, 120] → [B, 1000]


def build_data_batch(images, device):
    images = images.to(device)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
    Cmax, _ = images.max(dim=1); Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    H = torch.zeros_like(I_R)
    mr = (Cmax == I_R); mg = (Cmax == I_G) & ~mr; mb = ~mr & ~mg
    H[mr] = (((I_G[mr] - I_B[mr]) / delta[mr]) % 6)
    H[mg] = ((I_B[mg] - I_R[mg]) / delta[mg]) + 2
    H[mb] = ((I_R[mb] - I_G[mb]) / delta[mb]) + 4
    H = H / 6.0
    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))
    total = I_R + I_G + I_B + 1e-8
    return {
        'I_B': I_B, 'I_BY': I_B - (I_R + I_G) / 2,
        'I_G': I_G, 'I_GRAY': I_GRAY, 'I_H': H,
        'I_R': I_R, 'I_RG': I_R - I_G, 'I_S': S,
        'I_g': I_G / total, 'I_r': I_R / total,
    }


def main():
    torch.manual_seed(42)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading ImageNet (20/class, 112×112)...")
    dm = ImageNetDataModule(data_dir=DATA_DIR, resolution=112, batch_size=256,
                            num_workers=8, samples_per_class=20)
    dm.setup()
    train_loader = DataLoader(dm.train_dataset, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True)

    # Init kernel bank and pretrainer
    kb = SymbolicKernelBank(device=DEVICE)
    model = KernelPretrainer(kb).to(DEVICE)

    # Only optimize learnable kernels (conv3x3, conv5x5) and the classifier head
    # Classic kernels are frozen in KernelPretrainer.__init__
    optimizer = torch.optim.AdamW(
        [kb.conv3x3, kb.conv5x5] + list(model.fc.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print(f"Pretraining {model.n_learnable} learnable kernels × 10 terminals = {model.n_learnable * 10} features")
    print(f"Classic kernels (6) frozen — already Sobel/Gabor values")
    print(f"Training for 10 epochs on {len(dm.train_dataset)} images...")

    t0 = time.time()
    for epoch in range(10):
        model.train()
        total_loss = 0; correct = 0; total = 0
        for images, labels in train_loader:
            terminal_dict = build_data_batch(images, DEVICE)
            labels = labels.to(DEVICE)
            logits = model(terminal_dict)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/10: loss={total_loss/len(train_loader):.4f}, acc={acc:.1f}%")

    elapsed = time.time() - t0
    print(f"\nPretraining done in {elapsed:.0f}s")

    # Save pretrained kernel bank weights
    torch.save(kb.state_dict(), OUTPUT_PATH)
    print(f"Saved pretrained kernels to {OUTPUT_PATH}")

    # Show what the learnable kernels learned
    print(f"\nLearned conv3x3_0:\n{kb.conv3x3[0, 0].detach().cpu()}")
    print(f"Learned conv5x5_0 (center 3x3):\n{kb.conv5x5[0, 0, 1:4, 1:4].detach().cpu()}")
    print(f"Classic kernels unchanged (frozen during pretraining)")


if __name__ == '__main__':
    main()
