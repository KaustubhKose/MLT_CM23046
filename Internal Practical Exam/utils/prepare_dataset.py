"""
utils/prepare_dataset.py
════════════════════════
Helpers for assembling a fruit dataset:

1. download_from_urls()  — fetch images from a text file of URLs
2. split_dataset()       — split a flat folder into train/val sets
3. verify_images()       — remove corrupt or non-image files
4. dataset_stats()       — print class distribution

Quick start
-----------
    python utils/prepare_dataset.py --action split \
        --source raw_data/apple --class apple

or as a library:
    from utils.prepare_dataset import split_dataset
    split_dataset("raw_data", "data", val_ratio=0.2)
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional

# Optional: comment out if not installed
try:
    import requests
    from PIL import Image
    DEPS_OK = True
except ImportError:
    DEPS_OK = False


# ─────────────────────────────────────────────
# 1. DOWNLOAD
# ─────────────────────────────────────────────
def download_from_urls(url_file: str, dest_dir: str, class_name: str,
                       max_images: int = 500):
    """
    Download images listed one-per-line in `url_file` into
    dest_dir/<class_name>/.
    """
    if not DEPS_OK:
        raise ImportError("Install requests and Pillow: pip install requests pillow")

    dest = Path(dest_dir) / class_name
    dest.mkdir(parents=True, exist_ok=True)

    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    downloaded = 0
    for i, url in enumerate(urls[:max_images]):
        try:
            resp = requests.get(url, timeout=10)
            ext  = url.split(".")[-1].split("?")[0][:4]
            if ext.lower() not in ("jpg", "jpeg", "png", "webp"):
                ext = "jpg"
            fname = dest / f"{class_name}_{i:04d}.{ext}"
            fname.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            print(f"  ⚠ Skipped {url}: {e}")

    print(f"✅ Downloaded {downloaded} images → {dest}")


# ─────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────
def split_dataset(source_root: str, dest_root: str,
                  val_ratio: float = 0.20, seed: int = 42):
    """
    Expects source_root/<class>/*.jpg structure.
    Creates dest_root/train/<class>/ and dest_root/val/<class>/.
    """
    random.seed(seed)
    source_root = Path(source_root)
    dest_root   = Path(dest_root)

    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = list(class_dir.glob("*"))
        images = [p for p in images if p.suffix.lower() in
                  (".jpg", ".jpeg", ".png", ".webp", ".bmp")]

        random.shuffle(images)
        split  = int(len(images) * (1 - val_ratio))
        train_imgs = images[:split]
        val_imgs   = images[split:]

        for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = dest_root / subset / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, out_dir / img_path.name)

        print(f"  {class_name}: {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"\n✅ Dataset split complete → {dest_root}")


# ─────────────────────────────────────────────
# 3. VERIFY
# ─────────────────────────────────────────────
def verify_images(data_root: str) -> int:
    """Remove corrupt / non-image files. Returns count of removed files."""
    if not DEPS_OK:
        raise ImportError("Install Pillow: pip install pillow")

    removed = 0
    for path in Path(data_root).rglob("*"):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"  🗑  Removing corrupt file: {path}")
            path.unlink()
            removed += 1

    print(f"✅ Verification done. Removed {removed} corrupt file(s).")
    return removed


# ─────────────────────────────────────────────
# 4. STATS
# ─────────────────────────────────────────────
def dataset_stats(data_root: str):
    """Print per-class image counts for train and val splits."""
    data_root = Path(data_root)
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    print(f"\n📊 Dataset statistics — {data_root}\n")
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("  " + "─" * 42)

    totals = {"train": 0, "val": 0}
    classes = set()

    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                classes.add(cls_dir.name)

    for cls in sorted(classes):
        counts = {}
        for split in ("train", "val"):
            cls_dir = data_root / split / cls
            counts[split] = sum(
                1 for p in cls_dir.glob("*") if p.suffix.lower() in EXTS
            ) if cls_dir.exists() else 0
            totals[split] += counts[split]
        total = sum(counts.values())
        print(f"  {cls:<15} {counts['train']:>8} {counts['val']:>8} {total:>8}")

    print("  " + "─" * 42)
    grand = totals["train"] + totals["val"]
    print(f"  {'TOTAL':<15} {totals['train']:>8} {totals['val']:>8} {grand:>8}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["download", "split", "verify", "stats"],
                        required=True)
    parser.add_argument("--source",  default="raw_data")
    parser.add_argument("--dest",    default="data")
    parser.add_argument("--class",   dest="class_name", default=None)
    parser.add_argument("--urls",    default=None, help="Text file with image URLs")
    parser.add_argument("--val",     type=float,  default=0.2)
    args = parser.parse_args()

    if args.action == "download":
        download_from_urls(args.urls, args.source, args.class_name)
    elif args.action == "split":
        split_dataset(args.source, args.dest, val_ratio=args.val)
    elif args.action == "verify":
        verify_images(args.dest)
    elif args.action == "stats":
        dataset_stats(args.dest)
