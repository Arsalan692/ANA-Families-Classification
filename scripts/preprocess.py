"""
Phase 3 -- Preprocessing: 42,279 different-sized PNGs  ->  one fixed-size array

Each cell is placed, unchanged, in the centre of a 100x100 black canvas
("padding"). Nothing is resized and no pixel value is altered, so every cell
keeps its true scale and brightness. 100 is the largest width/height in the
dataset, so nothing is cut off. (Phase 2 showed image size does not reveal the
class, so the size of the black border is not a shortcut.)

All canvases are stacked into ONE file, because Colab loads one big file in
seconds but reading 42,279 small files from Drive is very slow.

Saved in data/cells3_100.npz (row i matches row i of manifest.csv):
  images      uint8  (N, 100, 100)   0 = black ... 255 = white
  labels      uint8  (N,)            class index 0, 1, 2
  file_index  int32  (N,)            number in the original file name
  heights     uint8  (N,)            original image height
  widths      uint8  (N,)            original image width
heights/widths say which part of each canvas is real image, so later steps
can ignore the black border when they measure brightness.

Run:  python scripts/preprocess.py
"""
import numpy as np
import pandas as pd
from PIL import Image

import config

SIZE = config.CANVAS_SIZE


def corner(height, width):
    """Top-left position that centres an image of this size on the canvas."""
    return (SIZE - height) // 2, (SIZE - width) // 2


def main():
    manifest = pd.read_csv(config.MANIFEST_PATH)
    n = len(manifest)

    # ------------------------------------------------ 1. pad every image onto a canvas
    images = np.zeros((n, SIZE, SIZE), dtype=np.uint8)      # starts all black
    heights = np.zeros(n, dtype=np.uint8)
    widths = np.zeros(n, dtype=np.uint8)
    print(f"Padding {n} images onto {SIZE}x{SIZE} canvases...")
    for i, path in enumerate(manifest["image_path"]):
        with Image.open(config.CELLS_DIR / path) as im:
            pixels = np.asarray(im)
        h, w = pixels.shape
        assert h <= SIZE and w <= SIZE, f"{path} is larger than the canvas: {w}x{h}"
        top, left = corner(h, w)
        images[i, top:top + h, left:left + w] = pixels
        heights[i], widths[i] = h, w

    # ------------------------------------------------ 2. save
    np.savez_compressed(
        config.PACKED_PATH,
        images=images,
        labels=manifest["label"].to_numpy(dtype=np.uint8),
        file_index=manifest["file_index"].to_numpy(dtype=np.int32),
        heights=heights,
        widths=widths,
    )

    # ------------------------------------------------ 3. verify by reloading
    # Read each array ONCE: an .npz unpacks the whole array every time it is indexed.
    with np.load(config.PACKED_PATH) as archive:
        packed = {key: archive[key] for key in archive.files}
    assert packed["images"].shape == (n, SIZE, SIZE) and packed["images"].dtype == np.uint8
    assert (packed["labels"] == manifest["label"].values).all(), "labels out of step with manifest"
    assert (packed["file_index"] == manifest["file_index"].values).all(), "rows out of step with manifest"

    # 500 random images: the real area must match the original PNG pixel for pixel,
    # and everything outside it must be pure black.
    rng = np.random.default_rng(config.SEED)
    for i in rng.choice(n, 500, replace=False):
        with Image.open(config.CELLS_DIR / manifest["image_path"].iloc[i]) as im:
            original = np.asarray(im)
        h, w = int(packed["heights"][i]), int(packed["widths"][i])
        top, left = corner(h, w)
        canvas = packed["images"][i]
        assert original.shape == (h, w)
        assert np.array_equal(canvas[top:top + h, left:left + w], original), f"pixels changed in row {i}"
        assert int(canvas.sum()) == int(original.sum()), f"border not black in row {i}"

    # ------------------------------------------------ 4. report
    size_mb = config.PACKED_PATH.stat().st_size / 1e6
    counts = np.bincount(packed["labels"], minlength=len(config.CLASS_NAMES))
    report = [
        f"Packed {n} images into {config.PACKED_PATH.name}",
        f"  array shape {packed['images'].shape}, type {packed['images'].dtype}",
        f"  per class: " + ", ".join(f"{name} {c}" for name, c in zip(config.CLASS_NAMES, counts)),
        f"  file size on disk {size_mb:.0f} MB; in memory {packed['images'].nbytes / 1e6:.0f} MB",
        f"  real image area covers {100 * (heights.astype(int) * widths).sum() / (n * SIZE * SIZE):.0f}% "
        f"of the canvases; the rest is black border",
        "  verified: labels and row order match the manifest; 500 random images are "
        "pixel-identical to the originals with a pure black border",
    ]
    (config.OUTPUTS_DIR / "phase3_preprocess_log.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
