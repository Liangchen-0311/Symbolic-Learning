"""
Download ImageNet (ILSVRC2012) from HuggingFace and convert to ImageFolder format.

Usage:
    python scripts/download_imagenet.py --split validation   # 6.7 GB, fast
    python scripts/download_imagenet.py --split train         # 147 GB, slow
    python scripts/download_imagenet.py --split all           # both
"""

import os
import sys
import argparse
from pathlib import Path

# ── Force all HF caches onto the large /workspace volume (81T free) ──
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
os.environ["HF_TOKEN"] = HF_TOKEN

from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm


def check_disk_space(path="/workspace", required_gb=200):
    """Abort early if the target volume doesn't have enough room."""
    stat = os.statvfs(path)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    print(f"[preflight] Free space on {path}: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"ERROR: Need at least {required_gb} GB free, only {free_gb:.1f} GB available.")
        sys.exit(1)
    print("[preflight] Disk space OK.\n")


# ImageNet class index → WordNet ID mapping (first 10 shown, full list built from dataset)
def save_split(dataset, split_name, output_dir):
    """Save a HF dataset split to ImageFolder format."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {split_name} split: {len(dataset)} images → {split_dir}")

    # Get class names from the dataset features
    class_names = dataset.features['label'].names  # list of 1000 WordNet IDs

    # Create class directories
    for cls_name in class_names:
        (split_dir / cls_name).mkdir(exist_ok=True)

    # Save images
    skipped = 0
    for i, example in enumerate(tqdm(dataset, desc=f"Saving {split_name}")):
        label = example['label']
        cls_name = class_names[label]
        img = example['image']

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_path = split_dir / cls_name / f"{i:08d}.JPEG"
        if not img_path.exists():
            try:
                img.save(img_path, 'JPEG', quality=95)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Warning: skipped image {i}: {e}")

    print(f"  Done: {len(dataset) - skipped} saved, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation', 'all'],
                        default='all')
    parser.add_argument('--output_dir', default='/workspace/neurosymbolic-rl/data/imagenet')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode (lower RAM but slower)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight
    check_disk_space("/workspace", required_gb=200)
    login(token=HF_TOKEN)
    print(f"Authenticated with HuggingFace. Cache dir: {os.environ['HF_DATASETS_CACHE']}\n")

    splits = ['train', 'validation'] if args.split == 'all' else [args.split]

    for split in splits:
        # Map 'validation' to 'val' directory for torchvision compatibility
        dir_name = 'val' if split == 'validation' else split

        target_dir = output_dir / dir_name
        if target_dir.exists() and len(list(target_dir.iterdir())) >= 999:
            print(f"\n{dir_name}/ already has {len(list(target_dir.iterdir()))} dirs, skipping download")
            continue

        print(f"\nLoading {split} from HuggingFace (this may take a while)...")

        if args.streaming:
            ds = load_dataset('ILSVRC/imagenet-1k', split=split, streaming=True)
            # For streaming, we need to iterate and save
            class_names = None
            split_dir = output_dir / dir_name
            split_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for example in tqdm(ds, desc=f"Streaming {split}"):
                if class_names is None:
                    # First iteration: we need to create dirs
                    # In streaming mode, features might not expose names easily
                    # We'll create dirs on-the-fly
                    pass

                label = example['label']
                cls_dir = split_dir / f"class_{label:04d}"
                cls_dir.mkdir(exist_ok=True)

                img = example['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(cls_dir / f"{count:08d}.JPEG", 'JPEG', quality=95)
                count += 1

            print(f"  Saved {count} images")
        else:
            ds = load_dataset('ILSVRC/imagenet-1k', split=split)
            save_split(ds, dir_name, output_dir)

    print(f"\nDone! Data saved to: {output_dir}")
    print("Directory structure:")
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            n_classes = len([x for x in d.iterdir() if x.is_dir()])
            n_imgs = sum(1 for _ in d.rglob('*.JPEG'))
            print(f"  {d.name}/: {n_classes} classes, {n_imgs} images")


if __name__ == '__main__':
    main()
