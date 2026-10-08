"""
Phase 1 -- Qi cells adapter: labels.mat + image folder  ->  clean data/manifest.csv

Pairs every label in labels.mat with its image file, keeps the classes listed
in config.CLASSES, checks every kept image, and writes our standard manifest:

    image_path, sample_id, label, source, label_name, file_index

This file is the ONLY place that knows this dataset's layout. Every later
phase reads manifest.csv and never looks at labels.mat again.

Notes on the columns
  image_path  relative to the dataset folder, e.g. 'cells/123.png'
  sample_id   empty: the dataset has no specimen IDs (filled in Phase 4 if a
              grouping is derived, or when real IDs are obtained)
  label       class index the model uses: 0, 1, 2 in the order of config.CLASSES
  file_index  the number in the file name; kept because file order may follow
              specimens (tested in Phase 2)

Outputs
  data/manifest.csv                 one row per image we will use
  outputs/phase1_cleaning_log.txt   what was checked, dropped and why

Run:  python scripts/build_manifest.py
"""
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

import config

# Counts seen when the dataset was first inspected (8 Oct 2026). A mismatch
# means the dataset folder changed and the run stops.
EXPECTED_COUNTS = {1: 14367, 2: 14655, 3: 13257, 4: 13737, 5: 5086, 6: 2343}


def main():
    log = []  # human-readable record of every check and drop

    # ------------------------------------------------ 1. read the labels
    mat = sio.loadmat(str(config.CELLS_LABELS_MAT))
    labels = mat["labels"].ravel().astype(int)   # entry i-1 = label of i.png
    n_total = len(labels)
    counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    log.append(f"labels.mat: {n_total} labels; per label {counts}")
    assert counts == EXPECTED_COUNTS, f"label counts changed: {counts}"

    # ------------------------------------------------ 2. compare labels with files on disk
    on_disk = {int(p.stem) for p in config.CELLS_IMG_DIR.glob("*.png")}
    wanted = set(range(1, n_total + 1))
    missing, extra = sorted(wanted - on_disk), sorted(on_disk - wanted)
    log.append(f"Image files on disk: {len(on_disk)}; labels without a file: {len(missing)}; "
               f"files without a label: {len(extra)}")
    assert not missing and not extra, f"files and labels disagree: {missing[:5]} {extra[:5]}"

    # ------------------------------------------------ 3. keep the chosen classes
    table = pd.DataFrame({"file_index": np.arange(1, n_total + 1), "orig_label": labels})
    table = table[table["orig_label"].isin(config.CLASSES)].reset_index(drop=True)
    log.append(f"Kept labels {list(config.CLASSES)} ({', '.join(config.CLASS_NAMES)}): "
               f"{len(table)} images; left out {n_total - len(table)} images of other classes")

    # ------------------------------------------------ 4. open every kept image
    # Fully decoding each file catches corrupt images now instead of mid-training.
    # The hash of the pixels finds exact duplicates even if the files were
    # saved differently.
    unreadable, not_grey = [], []
    by_hash = defaultdict(list)
    for idx in table["file_index"]:
        path = config.CELLS_IMG_DIR / f"{idx}.png"
        try:
            with Image.open(path) as im:
                if im.mode != "L":               # 'L' = 8-bit greyscale
                    not_grey.append(idx)
                pixels = np.asarray(im)
            by_hash[hashlib.md5(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()].append(idx)
        except Exception as error:               # noqa: BLE001 - any failure means "unreadable"
            unreadable.append(idx)
            log.append(f"Unreadable, dropped: {idx}.png ({error})")
    log.append(f"Opened {len(table)} images: unreadable {len(unreadable)}, not greyscale {len(not_grey)}")
    assert not not_grey, f"unexpected colour images: {not_grey[:5]}"
    table = table[~table["file_index"].isin(unreadable)]

    # ------------------------------------------------ 5. exact duplicates
    # Same pixels, same class -> keep the first copy.
    # Same pixels, different classes -> the label cannot be trusted, drop all.
    label_of = dict(zip(table["file_index"], table["orig_label"]))
    drop = set()
    groups = [g for g in by_hash.values() if len(g) > 1]
    for group in groups:
        if len({label_of[i] for i in group}) == 1:
            drop.update(group[1:])
            log.append(f"Duplicate pixels, kept {group[0]}.png, dropped {group[1:]}")
        else:
            drop.update(group)
            log.append(f"Duplicate pixels with conflicting labels, dropped all: {group}")
    log.append(f"Exact-duplicate groups: {len(groups)}; images dropped: {len(drop)}")
    table = table[~table["file_index"].isin(drop)].reset_index(drop=True)

    # ------------------------------------------------ 6. build the standard manifest
    manifest = pd.DataFrame({
        "image_path": "cells/" + table["file_index"].astype(str) + ".png",
        "sample_id": "",                                   # unknown for this dataset
        "label": table["orig_label"].map(config.LABEL_TO_INDEX),
        "source": config.SOURCE,
        "label_name": table["orig_label"].map(config.CLASSES),
        "file_index": table["file_index"],
    })[config.MANIFEST_COLUMNS + config.EXTRA_COLUMNS]

    # ------------------------------------------------ 7. sanity checks (stop if wrong)
    assert manifest["label"].isin(range(len(config.CLASSES))).all(), "unknown label value"
    assert manifest["image_path"].is_unique, "two rows point to the same image"
    assert manifest["file_index"].is_monotonic_increasing, "rows are not in file order"
    assert all((config.CELLS_DIR / p).is_file() for p in manifest["image_path"])
    assert (manifest["label_name"] == manifest["label"].map(dict(enumerate(config.CLASS_NAMES)))).all()

    # ------------------------------------------------ 8. save
    config.DATA_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    manifest.to_csv(config.MANIFEST_PATH, index=False)

    log += ["", "================ SUMMARY ================", f"Images kept : {len(manifest)}"]
    for index, name in enumerate(config.CLASS_NAMES):
        rows = manifest[manifest["label"] == index]
        log.append(f"  label {index}  {name:<12} {len(rows):>6} images   "
                   f"file numbers {rows['file_index'].min()} to {rows['file_index'].max()}")
    log.append("Specimen IDs: none in this dataset (sample_id left empty)")
    log_path = config.OUTPUTS_DIR / "phase1_cleaning_log.txt"
    log_path.write_text("\n".join(log), encoding="utf-8")

    print("\n".join(log))
    print(f"\nSaved: {config.MANIFEST_PATH}\nSaved: {log_path}")


if __name__ == "__main__":
    main()
