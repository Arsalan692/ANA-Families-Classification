"""
Phase 4 -- Splitting: find slide-like blocks in the file order, then make 5 folds

The dataset has no specimen IDs, but Phase 2 showed the files were saved in
groups that belong together (brightness sits on a flat level, then jumps).
This script:

  1. finds those jumps inside each class            -> "blocks" of consecutive files
  2. gives every block a name, stored in sample_id  -> a stand-in for the missing
                                                       specimen ID (derived, not real)
  3. deals whole blocks into 5 folds, keeping the mix of classes and of
     dim / bright blocks the same in every fold     -> 'fold' column in the manifest

A block is never divided, so neighbouring files always share a fold.

How a jump is found: at every file position, compare the median of the 30
files before with the median of the 30 files after. A cut is made where that
difference is at least JUMP times the normal cell-to-cell variation. The
threshold is deliberately high: missing a boundary only merges two slides
(harmless), while a false boundary could put one slide in two folds (leakage).

Needs outputs/eda/image_stats.csv from Phase 2.
Outputs
  data/manifest.csv                      + sample_id filled, + fold column
  outputs/folds/fold_table.txt           counts and checks
  outputs/folds/blocks_and_folds.png     picture of the blocks and their folds

Run:  python scripts/make_folds.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold, cross_val_predict

import config

OUT = config.OUTPUTS_DIR / "folds"
STATS_PATH = config.OUTPUTS_DIR / "eda" / "image_stats.csv"
WINDOW = 30                    # files compared on each side of a candidate cut
JUMP = 3.0                     # cut where the level changes by >= 3x the cell-to-cell variation
FEATURES = ["mean", "std"]     # brightness level and contrast of each cell
COLOR = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"


def jump_score(part):
    """For each file (in order): how big is the level change at this position?"""
    score = np.zeros(len(part))
    for col in FEATURES:
        x = np.log(part[col])                                   # log: a 10% change counts the same when dim or bright
        level = x.rolling(WINDOW).median()
        before, after = level.shift(1), level.shift(-(WINDOW - 1))
        variation = 1.4826 * (x - x.rolling(2 * WINDOW + 1, center=True).median()).abs().median()
        score = np.maximum(score, ((after - before).abs() / variation).fillna(0).values)
    return score


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config.MANIFEST_PATH)
    stats = pd.read_csv(STATS_PATH)
    assert (stats["file_index"].values == manifest["file_index"].values).all(), "re-run scripts/eda.py --remeasure"
    df = manifest.drop(columns=[c for c in ["fold"] if c in manifest]).copy()
    df[FEATURES] = stats[FEATURES].values
    report = []

    # ------------------------------------------------ 1. find blocks inside each class
    df["block"] = -1
    cuts = {}
    for k, name in enumerate(config.CLASS_NAMES):
        rows = np.flatnonzero(df["label"].values == k)           # manifest is in file order
        peaks, _ = find_peaks(jump_score(df.iloc[rows]), height=JUMP, distance=WINDOW)
        block_of = np.searchsorted(peaks, np.arange(len(rows)), side="right")
        df.iloc[rows, df.columns.get_loc("block")] = block_of
        cuts[k] = df["file_index"].values[rows][peaks]
        sizes = np.bincount(block_of)
        report.append(f"{name:<12} {len(sizes):>3} blocks; files per block: smallest {sizes.min()}, "
                      f"typical {int(np.median(sizes))}, largest {sizes.max()}")
    df["sample_id"] = (config.SOURCE + "_b" + df["label"].astype(str) + "_"
                       + df["block"].astype(str).str.zfill(3))

    # ------------------------------------------------ 2. deal blocks into folds
    # Every block is tagged dim or bright (against the typical block of its class),
    # so each fold receives its share of both kinds from every class.
    block_level = df.groupby("sample_id")["mean"].median()
    block_label = df.groupby("sample_id")["label"].first()
    bright = block_level > block_level.groupby(block_label).transform("median")
    stratum = (df["label"] * 2 + df["sample_id"].map(bright).astype(int)).values
    splitter = StratifiedGroupKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    df["fold"] = -1
    for fold, (_, test_rows) in enumerate(splitter.split(df, stratum, groups=df["sample_id"])):
        df.iloc[test_rows, df.columns.get_loc("fold")] = fold

    # ------------------------------------------------ 3. checks (stop if wrong)
    assert df["fold"].between(0, config.N_FOLDS - 1).all(), "an image has no fold"
    assert (df.groupby("sample_id")["fold"].nunique() == 1).all(), "a block is in two folds"
    assert (df.groupby("sample_id")["label"].nunique() == 1).all(), "a block mixes classes"
    for _, g in df.groupby("sample_id"):                         # blocks are unbroken runs of file numbers
        assert g["file_index"].max() - g["file_index"].min() + 1 == len(g), "a block is not consecutive"

    # ------------------------------------------------ 4. tables
    names = df["label"].map(dict(enumerate(config.CLASS_NAMES)))
    images = pd.crosstab(df["fold"], names)[config.CLASS_NAMES]
    images["total"] = images.sum(axis=1)
    blocks = pd.crosstab(df.drop_duplicates("sample_id")["fold"],
                         names[df.drop_duplicates("sample_id").index])[config.CLASS_NAMES]
    level = df.pivot_table(index="fold", columns=names, values="mean", aggfunc="median")[config.CLASS_NAMES]
    share = (100 * images[config.CLASS_NAMES].div(images["total"], axis=0)).round(1)
    report += ["", "IMAGES per fold:", images.to_string(), "",
               "CLASS MIX per fold (% of the fold):", share.to_string(), "",
               "BLOCKS per fold:", blocks.to_string(), "",
               "TYPICAL BRIGHTNESS per fold (median of average brightness):", level.round(1).to_string(), ""]

    # How each of the 5 rounds uses the folds
    report.append("ROUNDS: test = fold k, validation = fold k+1, training = the other three")
    for k in range(config.N_FOLDS):
        val = (k + 1) % config.N_FOLDS
        n_test, n_val = int((df.fold == k).sum()), int((df.fold == val).sum())
        report.append(f"  round {k}: test fold {k} ({n_test}), validation fold {val} ({n_val}), "
                      f"training {len(df) - n_test - n_val}")
    report.append("")

    # Leakage reference from Phase 2: a classifier that sees only six brightness
    # numbers scored 0.864 on random folds and 0.699 on plain consecutive fifths.
    six = stats[["mean", "std", "p05", "p50", "p99", "max"]].values
    pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=config.SEED),
                             six, df["label"].values, cv=PredefinedSplit(df["fold"].values))
    report.append(f"Brightness-only classifier on these folds: balanced accuracy "
                  f"{balanced_accuracy_score(df['label'], pred):.3f} "
                  f"(Phase 2: 0.864 random folds, 0.699 consecutive fifths)")

    # ------------------------------------------------ 5. picture
    fig, axes = plt.subplots(2 * len(config.CLASS_NAMES), 1, figsize=(11, 9), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [4, 1.3] * len(config.CLASS_NAMES)})
    top = np.percentile(df["mean"], 99.8)
    for k, name in enumerate(config.CLASS_NAMES):
        part = df[df.label == k]
        ax, strip = axes[2 * k], axes[2 * k + 1]
        ax.vlines(cuts[k], 0, top, color="#c3c2b7", linewidth=0.6)
        ax.scatter(part["file_index"], part["mean"], s=1.2, color=COLOR[k], alpha=0.4, linewidths=0)
        ax.set_title(f"{name}: brightness in file order, grey lines = detected block boundaries "
                     f"({part['sample_id'].nunique()} blocks)", loc="left", color=INK, fontsize=10, pad=6)
        ax.set_ylabel("brightness", color=INK_2, fontsize=8)
        ax.set_ylim(0, top)
        ax.set_xticks([])
        spans = part.groupby("sample_id").agg(first=("file_index", "min"), last=("file_index", "max"),
                                              fold=("fold", "first"))
        strip.hlines(spans["fold"], spans["first"], spans["last"], color=INK, linewidth=3)
        strip.set_yticks(range(config.N_FOLDS))
        strip.set_ylim(config.N_FOLDS - 0.4, -0.6)
        strip.set_ylabel("fold", color=INK_2, fontsize=8)
        strip.grid(axis="y", color=GRID, linewidth=0.6)
        for a in (ax, strip):
            a.set_facecolor(SURFACE)
            a.set_xlim(part["file_index"].min() - 20, part["file_index"].max() + 20)
            a.tick_params(colors=INK_2, labelsize=7)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color("#c3c2b7")
    axes[-1].set_xlabel("file number   (each bar in a 'fold' strip is one block; its row is the fold it went to)",
                        color=INK_2, fontsize=8)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(OUT / "blocks_and_folds.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ 6. save
    columns = config.MANIFEST_COLUMNS + config.EXTRA_COLUMNS + ["fold"]
    df[columns].to_csv(config.MANIFEST_PATH, index=False)
    (OUT / "fold_table.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved: {config.MANIFEST_PATH}\nSaved: {OUT}")


if __name__ == "__main__":
    main()
