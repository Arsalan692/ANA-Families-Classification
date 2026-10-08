"""
Central settings for the 3-family ANA pattern classifier (cell level).

Every other script imports its paths and settings from here, so if a folder
moves or a setting changes we edit ONE file instead of hunting through many.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Project root = the "Model Training on Basic Classes" folder (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw Qi et al. cell dataset (read-only: we never modify these files)
CELLS_DIR = PROJECT_ROOT.parent.parent / "datasets" / "cells"
CELLS_IMG_DIR = CELLS_DIR / "cells"         # 1.png ... 63445.png
CELLS_LABELS_MAT = CELLS_DIR / "labels.mat" # entry i = label of i.png

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"   # clean label table (Phase 1)
PACKED_PATH = DATA_DIR / "cells3_100.npz"   # all images in one array (Phase 3)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"      # logs, tables, charts

# ---------------------------------------------------------------- classes
# Dataset label -> family name. The dataset gives numbers only; the names were
# read from sample images (8 Oct 2026). To add a class later, add a line here.
CLASSES = {
    1: "Homogeneous",   # AC-1
    2: "Speckled",      # AC-4, AC-5 (merged in this dataset)
    3: "Nucleolar",     # AC-8, AC-9, AC-10 (merged in this dataset)
}
CLASS_NAMES = list(CLASSES.values())                           # model output order
LABEL_TO_INDEX = {lab: i for i, lab in enumerate(CLASSES)}     # 1,2,3 -> 0,1,2
SOURCE = "QiCells"

# ---------------------------------------------------------------- settings
SEED = 42              # fixed random seed -> reproducible splits/training
N_FOLDS = 5            # 5-fold cross-validation
CANVAS_SIZE = 100      # each cell is centred on a 100x100 black canvas (no resize)
CROP_SIZE = 88         # centre region of the canvas that the network sees
INPUT_SIZE = 176       # the crop is enlarged 2x before it enters the network

# Standard manifest columns -- identical for every dataset (design rule),
# followed by the extras this dataset needs.
MANIFEST_COLUMNS = ["image_path", "sample_id", "label", "source"]
EXTRA_COLUMNS = ["label_name", "file_index"]
