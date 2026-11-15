"""Data acquisition + preprocessing for the bean leaf disease pipeline.

Steps applied to every image (training, uploaded retraining data and prediction input):
  1. decode the file and force 3-channel RGB (drops alpha / converts greyscale)
  2. resize to IMG_SIZE x IMG_SIZE (bilinear)
  3. keep pixel values as float32 in [0, 255]; the model itself rescales to [-1, 1]
     (MobileNetV2 convention) so the exact same function serves train and inference
Training additionally uses random augmentation layers that live inside the model
(active only while training).
"""
from __future__ import annotations

import io
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import config as C

DATASET_URL = "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/{split}.zip"


# ---------------------------------------------------------------- acquisition
def download_dataset(dest: Path = C.DATA_DIR) -> None:
    """Download the iBeans dataset (Makerere AI Lab, Uganda) into data/train and data/test.

    The original validation split is merged into train; a validation split is
    re-created from train at training time (VAL_SPLIT).
    """
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        zpath = raw / f"{split}.zip"
        if not zpath.exists():
            print(f"Downloading {split}...")
            urllib.request.urlretrieve(DATASET_URL.format(split=split), zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
    for split, target in (("train", C.TRAIN_DIR), ("validation", C.TRAIN_DIR), ("test", C.TEST_DIR)):
        for cls in C.CLASS_NAMES:
            (target / cls).mkdir(parents=True, exist_ok=True)
            for f in (raw / split / cls).glob("*"):
                shutil.move(str(f), target / cls / f.name)
    shutil.rmtree(raw, ignore_errors=True)


def count_images(directory: Path) -> dict[str, int]:
    return {
        cls: sum(1 for f in (directory / cls).glob("*") if f.suffix.lower() in C.ALLOWED_EXT)
        if (directory / cls).exists() else 0
        for cls in C.CLASS_NAMES
    }


def list_images(directory: Path) -> list[tuple[Path, int]]:
    """Return (path, class_index) pairs for an ImageFolder-style directory."""
    items = []
    for idx, cls in enumerate(C.CLASS_NAMES):
        for f in sorted((directory / cls).glob("*")):
            if f.suffix.lower() in C.ALLOWED_EXT:
                items.append((f, idx))
    return items


# ---------------------------------------------------------------- preprocessing
def load_image(source, size: int = C.IMG_SIZE) -> np.ndarray:
    """Decode a path / bytes / file-like object into a (size, size, 3) float32 array in [0, 255]."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)  # respect phone-camera orientation
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def validate_image(data: bytes) -> bool:
    """True if the bytes decode as an image (used to reject bad uploads)."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def load_arrays(items: list[tuple[Path, int]], size: int = C.IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a list of (path, label) into stacked X (N, size, size, 3) uint8 and y (N,).

    uint8 storage keeps memory 4x lower; batches are cast to float32 in make_dataset.
    """
    X = np.zeros((len(items), size, size, 3), dtype=np.uint8)
    y = np.zeros(len(items), dtype=np.int32)
    for i, (path, label) in enumerate(items):
        X[i] = load_image(path, size).astype(np.uint8)
        y[i] = label
    return X, y


def stratified_split(items, val_fraction=C.VAL_SPLIT, seed=C.SEED):
    """Split (path, label) items into train/val keeping the class balance."""
    rng = random.Random(seed)
    train, val = [], []
    for idx in range(len(C.CLASS_NAMES)):
        cls_items = [it for it in items if it[1] == idx]
        rng.shuffle(cls_items)
        n_val = max(1, int(len(cls_items) * val_fraction)) if cls_items else 0
        val += cls_items[:n_val]
        train += cls_items[n_val:]
    rng.shuffle(train)
    return train, val


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size=C.BATCH_SIZE, shuffle=False):
    """Wrap arrays in a batched, prefetched tf.data pipeline with one-hot labels."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, tf.one_hot(y, len(C.CLASS_NAMES))))
    if shuffle:
        ds = ds.shuffle(len(X), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda x, t: (tf.cast(x, tf.float32), t), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights (keeps retraining fair if uploads are imbalanced)."""
    counts = np.bincount(y, minlength=len(C.CLASS_NAMES)).astype(float)
    counts[counts == 0] = 1
    w = len(y) / (len(C.CLASS_NAMES) * counts)
    return {i: float(v) for i, v in enumerate(w)}


# ---------------------------------------------------------------- feature extraction (for insights)
def image_features(path: Path) -> dict:
    """Hand-crafted, interpretable colour/texture features used for the data-insight charts.

    - green_ratio:  share of pixels where green dominates (healthy leaf tissue)
    - lesion_ratio: share of brown/yellow/rust-coloured pixels (HSV hue 5-35 deg, saturated)
    - brightness:   mean V channel (lighting conditions)
    - saturation:   mean S channel
    - edge_density: share of strong-gradient pixels (spots/pustules create texture)
    """
    img = Image.open(path).convert("RGB").resize((128, 128))
    rgb = np.asarray(img, dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (g > r * 1.05) & (g > b * 1.05) & (v > 0.15)
    lesion = (h >= 5) & (h <= 35) & (s > 0.35) & (v > 0.2)
    grey = np.asarray(img.convert("L"), dtype=np.float32)
    gx, gy = np.abs(np.diff(grey, axis=1))[:-1], np.abs(np.diff(grey, axis=0))[:, :-1]
    edges = (gx + gy) > 40
    return {
        "green_ratio": float(green.mean()),
        "lesion_ratio": float(lesion.mean()),
        "brightness": float(v.mean()),
        "saturation": float(s.mean()),
        "edge_density": float(edges.mean()),
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
    }


def build_feature_table(directories=(C.TRAIN_DIR, C.TEST_DIR), out_csv: Path = C.FEATURES_CSV):
    import pandas as pd

    rows = []
    for d in directories:
        for path, idx in list_images(d):
            rows.append({"split": d.name, "class": C.CLASS_NAMES[idx], "file": path.name, **image_features(path)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
