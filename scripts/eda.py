"""
Phase 2 -- Exploratory Data Analysis (EDA)

Looks at the data BEFORE any model is trained, to answer three questions:
  1. Is there a SIZE shortcut?       (can the class be guessed from width/height?)
  2. Is there a BRIGHTNESS shortcut? (can it be guessed from simple brightness numbers?)
  3. Does FILE ORDER follow specimens? (are neighbouring files alike?)  -> decides
     how Phase 4 splits the data, because the dataset has no specimen IDs.

Uses only manifest.csv + the image pixels.
Outputs go to outputs/eda/ (image sheets go to outputs/eda/samples/, git-ignored).
Run:  python scripts/eda.py            (add --remeasure to re-read every image)
"""
import sys

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, cross_val_predict

import config

OUT = config.OUTPUTS_DIR / "eda"
SAMPLES = OUT / "samples"
STATS_PATH = OUT / "image_stats.csv"
NAMES = config.CLASS_NAMES
COLOR = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3, fixed order
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
LAGS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def image_stats(image_path):
    """Open one cell image and measure its size and brightness (0 = black, 255 = white)."""
    with Image.open(config.CELLS_DIR / image_path) as im:
        px = np.asarray(im).astype(np.float32)
    p05, p50, p99 = np.percentile(px, [5, 50, 99])
    return {"width": px.shape[1], "height": px.shape[0],
            "mean": px.mean(), "std": px.std(),
            "p05": p05,            # background level (the darkest part)
            "p50": p50, "p99": p99, "max": px.max()}


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def block_folds(df, n_folds):
    """Fold = which consecutive fifth of its class a file falls in (by file number)."""
    fold = np.zeros(len(df), dtype=int)
    for label in sorted(df["label"].unique()):
        idx = np.flatnonzero(df["label"].values == label)          # already in file order
        fold[idx] = np.arange(len(idx)) * n_folds // len(idx)
    return fold


def guess_accuracy(df, columns, folds):
    """Balanced accuracy of a simple classifier that sees ONLY the given columns."""
    model = HistGradientBoostingClassifier(max_iter=150, random_state=config.SEED)
    pred = cross_val_predict(model, df[columns].values, df["label"].values, cv=folds)
    return balanced_accuracy_score(df["label"], pred)


def autocorrelation(x, lag):
    """How alike are values `lag` files apart?  1 = identical trend, 0 = unrelated."""
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).mean() / x.var())


def sheet(df_rows, path, title):
    """Save a grid of cells. Brightness is stretched FOR VIEWING ONLY (raw cells are dim)."""
    cols = 12
    rows = int(np.ceil(len(df_rows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.4), facecolor="#1a1a19")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, row) in zip(np.ravel(axes), df_rows.iterrows()):
        with Image.open(config.CELLS_DIR / row["image_path"]) as im:
            ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=max(float(row["p99"]), 1.0))
        ax.set_title(f"#{row['file_index']}  {row['width']}x{row['height']}\nmean {row['mean']:.0f}",
                     fontsize=6, color="#c3c2b7")
    fig.suptitle(title + "   (display brightened for viewing only)", color="#ffffff", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config.MANIFEST_PATH)
    report = []

    # ---------------------------------------------------------- measure every image
    if STATS_PATH.exists() and "--remeasure" not in sys.argv:
        df = pd.read_csv(STATS_PATH)
        assert len(df) == len(manifest), "image_stats.csv is stale: run with --remeasure"
    else:
        print(f"Measuring {len(manifest)} images (about a minute)...")
        stats = pd.DataFrame([image_stats(p) for p in manifest["image_path"]])
        df = pd.concat([manifest, stats], axis=1)
        df.to_csv(STATS_PATH, index=False)
    df["area"] = df["width"] * df["height"]
    df["aspect"] = df["width"] / df["height"]
    name = df["label"].map(dict(enumerate(NAMES)))

    # ---------------------------------------------------------- 1. size
    report += ["IMAGE SIZE per class (pixels):",
               df.groupby(name)[["width", "height", "aspect"]].describe()
                 .loc[NAMES, (slice(None), ["mean", "min", "50%", "max"])].round(2).to_string(), ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    bins = np.arange(61.5, 101.5, 1)
    for ax, col in zip(axes, ["width", "height"]):
        for k, n in enumerate(NAMES):
            ax.hist(df.loc[df.label == k, col], bins=bins, density=True, histtype="step",
                    linewidth=2, color=COLOR[k], label=n)
        style(ax, f"Image {col} per class")
        ax.set_xlabel(f"{col} in pixels", color=INK_2, fontsize=9)
        ax.set_ylabel("share of images (density)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "size_hist.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 2. brightness
    report += ["BRIGHTNESS per class (0 = black, 255 = white):",
               df.groupby(name)[["mean", "p05", "p99", "max"]].describe()
                 .loc[NAMES, (slice(None), ["mean", "min", "50%", "max"])].round(1).to_string(), ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, col, title, top in [(axes[0], "mean", "Average brightness per image", 100),
                                (axes[1], "p99", "Brightest-spot level (99th percentile)", 255)]:
        bins = np.linspace(0, top, 61)
        for k, n in enumerate(NAMES):
            ax.hist(df.loc[df.label == k, col].clip(upper=top), bins=bins, density=True,
                    histtype="step", linewidth=2, color=COLOR[k], label=n)
        style(ax, title)
        ax.set_xlabel("pixel value (0 = black, 255 = full brightness)", color=INK_2, fontsize=9)
        ax.set_ylabel("share of images (density)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "brightness_hist.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 3. shortcut tests
    # A simple classifier is shown ONLY size or ONLY brightness numbers. Chance for
    # three classes is 0.333. "Random folds" mixes neighbouring files across train
    # and test; "block folds" keeps consecutive files together.
    random_folds = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    blocks = PredefinedSplit(block_folds(df, config.N_FOLDS))
    tests = [("size only (width, height, area, aspect)", ["width", "height", "area", "aspect"]),
             ("average brightness only", ["mean"]),
             ("six brightness numbers (mean, std, p05, p50, p99, max)",
              ["mean", "std", "p05", "p50", "p99", "max"])]
    report.append("SHORTCUT TESTS: balanced accuracy of a simple classifier (chance = 0.333)")
    report.append(f"  {'what the classifier sees':<56} random folds   block folds")
    for label, columns in tests:
        a, b = guess_accuracy(df, columns, random_folds), guess_accuracy(df, columns, blocks)
        report.append(f"  {label:<56} {a:>10.3f} {b:>13.3f}")
    report.append("")

    # ---------------------------------------------------------- 4. does file order follow specimens?
    report.append("FILE-ORDER TEST: are neighbouring files more alike than far-apart ones?")
    rng = np.random.default_rng(config.SEED)
    acf = {}
    for k, n in enumerate(NAMES):
        part = df[df.label == k].sort_values("file_index")
        row = [n]
        for col in ["mean", "p05"]:
            x = part[col].values
            neighbour = np.median(np.abs(np.diff(x)))
            random_pair = np.median(np.abs(x - rng.permutation(x)))
            row.append(f"{col}: neighbour gap {neighbour:.2f} vs random-pair gap {random_pair:.2f} "
                       f"(ratio {neighbour / random_pair:.2f})")
        acf[n] = [autocorrelation(part["mean"].values, lag) for lag in LAGS]
        report.append("  " + " | ".join(row))
    report.append("  Autocorrelation of average brightness along file order "
                  "(1 = neighbours identical, 0 = unrelated):")
    report.append("  " + f"{'lag (files apart)':<18}" + "".join(f"{lag:>7}" for lag in LAGS))
    for n in NAMES:
        report.append("  " + f"{n:<18}" + "".join(f"{v:>7.2f}" for v in acf[n]))
    report.append("")

    fig, axes = plt.subplots(len(NAMES), 1, figsize=(11, 7.5), facecolor=SURFACE)
    for k, (ax, n) in enumerate(zip(axes, NAMES)):
        part = df[df.label == k].sort_values("file_index")
        ax.scatter(part["file_index"], part["mean"], s=1.2, color=COLOR[k], alpha=0.35, linewidths=0)
        ax.plot(part["file_index"], part["mean"].rolling(51, center=True).median(),
                color=INK, linewidth=1.2, label="rolling median (51 files)")
        style(ax, f"{n}: average brightness of each cell, in file order")
        ax.set_ylabel("average brightness", color=INK_2, fontsize=9)
        ax.set_ylim(0, np.percentile(df["mean"], 99.8))
        ax.margins(x=0.005)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    axes[-1].set_xlabel("file number", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "file_order_brightness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    for k, n in enumerate(NAMES):
        ax.plot(LAGS, acf[n], color=COLOR[k], linewidth=2, marker="o", markersize=5, label=n)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xscale("log")
    style(ax, "How alike are cells that are N files apart?")
    ax.set_xlabel("N = distance in file numbers (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("autocorrelation of brightness", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "file_order_autocorrelation.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 5. image sheets (git-ignored)
    for k, n in enumerate(NAMES):
        part = df[df.label == k]
        sheet(part.iloc[rng.choice(len(part), 36, replace=False)].sort_values("file_index"),
              SAMPLES / f"random_{n}.png", f"{n}: 36 random cells")
    sheet(df.nsmallest(36, "p99"), SAMPLES / "extreme_darkest.png", "Darkest cells (lowest bright-spot level)")
    sheet(df.nlargest(36, "mean"), SAMPLES / "extreme_brightest.png", "Brightest cells")
    sheet(df.nsmallest(36, "std"), SAMPLES / "extreme_flattest.png", "Flattest cells (least contrast)")
    first = df[df.label == 0].sort_values("file_index").iloc[:72]
    sheet(first, SAMPLES / "consecutive_Homogeneous_first72.png", "Homogeneous: the first 72 files in order")

    (OUT / "eda_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nFigures and tables saved in {OUT}")


if __name__ == "__main__":
    main()
