"""
Monitor ImageNet download progress in real-time.

Usage:
    python scripts/monitor_imagenet.py
"""

import os
import time
import shutil

DATA_DIR = "/workspace/neurosymbolic-rl/data/imagenet"
CACHE_DIR = "/workspace/.cache/huggingface"

# ImageNet approximate sizes
SPLITS = {
    "train": {"expected_images": 1_281_167, "expected_gb": 147.0},
    "val":   {"expected_images": 50_000,    "expected_gb": 6.3},
}
TOTAL_EXPECTED_GB = 147.0 + 6.3  # train + val images
CACHE_EXPECTED_GB = 160.0         # HF cache (compressed downloads)


def count_images(path):
    """Count JPEG files under a directory."""
    count = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".JPEG"):
                count += 1
    return count


def get_dir_size_gb(path):
    """Get directory size in GB."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 3)


def progress_bar(current, total, width=40, label=""):
    pct = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label} |{bar}| {pct*100:5.1f}%"


def main():
    print("ImageNet Download Monitor")
    print("=" * 60)
    print(f"Data dir : {DATA_DIR}")
    print(f"Cache dir: {CACHE_DIR}")
    print("Press Ctrl+C to stop monitoring.\n")

    prev_size = 0
    prev_time = time.time()

    while True:
        # Disk usage
        data_gb = get_dir_size_gb(DATA_DIR)
        cache_gb = get_dir_size_gb(CACHE_DIR)
        total_gb = data_gb + cache_gb

        # Per-split image counts
        train_dir = os.path.join(DATA_DIR, "train")
        val_dir = os.path.join(DATA_DIR, "val")
        train_imgs = count_images(train_dir) if os.path.exists(train_dir) else 0
        val_imgs = count_images(val_dir) if os.path.exists(val_dir) else 0

        # Download speed
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            speed_mb = (total_gb - prev_size) * 1024 / dt  # MB/s
        else:
            speed_mb = 0
        prev_size = total_gb
        prev_time = now

        # Free space
        stat = os.statvfs("/workspace")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

        # Check if process is still running
        import subprocess
        ret = subprocess.run(["pgrep", "-f", "download_imagenet"], capture_output=True)
        running = ret.returncode == 0

        # Display
        os.system("clear")
        print("╔══════════════════════════════════════════════════════════╗")
        print("║           ImageNet Download Progress Monitor            ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Status: {'🟢 DOWNLOADING' if running else '🔴 STOPPED'}                                    ║")
        print("╠══════════════════════════════════════════════════════════╣")

        # Train progress
        print(f"║  Train images: {train_imgs:>8,} / {SPLITS['train']['expected_images']:>10,}          ║")
        print(f"║  {progress_bar(train_imgs, SPLITS['train']['expected_images'], 44, '')}  ║")

        # Val progress
        print(f"║  Val images:   {val_imgs:>8,} / {SPLITS['val']['expected_images']:>10,}          ║")
        print(f"║  {progress_bar(val_imgs, SPLITS['val']['expected_images'], 44, '')}  ║")

        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Data size:  {data_gb:>8.2f} GB / ~{TOTAL_EXPECTED_GB:.0f} GB                    ║")
        print(f"║  Cache size: {cache_gb:>8.2f} GB                                ║")
        print(f"║  Total used: {total_gb:>8.2f} GB                                ║")
        print(f"║  Speed:      {speed_mb:>8.1f} MB/s                              ║")
        print(f"║  Free space: {free_gb:>8.1f} GB on /workspace                   ║")
        print("╚══════════════════════════════════════════════════════════╝")

        if not running:
            print("\nDownload process has stopped. Check log:")
            print(f"  tail -20 /workspace/neurosymbolic-rl/logs/imagenet_download.log")
            break

        time.sleep(10)


if __name__ == "__main__":
    main()
