"""
Phase 9: predict the ANA pattern family of ONE cropped cell image.

    python scripts/predict_pattern.py cell.png                 # one image
    python scripts/predict_pattern.py a.png b.png --json       # several, machine-readable output
    python scripts/predict_pattern.py --self-test              # check against the saved test predictions

From another program (the web app):

    from predict_pattern import PatternPredictor
    predictor = PatternPredictor()            # loads the models once, at start-up
    result = predictor.predict("cell.png")    # {"family": ..., "confidence": ..., "probabilities": {...}, ...}

This file is deliberately self-contained (it does not import config.py), so the
web app needs only this file plus the model folder.

The model folder (default: outputs/resnet18_adjusted) must hold, per round k:
    round_k/best.pt        trained weights
    round_k/metrics.json   the mean/std used to normalise the pixels in that round
and calibration/temperatures.json (Phase 7).

What happens to an image, in order (steps 3-6 are exactly what training did):
  1. Colour -> grey. The brightest colour channel is kept: green for a green
     image, the image itself for a greyscale one.
  2. Images larger than 100 px are shrunk to fit 100 px. The training cells were
     62 to 100 px, so this brings a larger crop to a similar scale.
  3. The cell is centred on a 100x100 black canvas.
  4. Brightness stretch ("adjusted" mode): background -> black, bright spots -> white.
  5. The centre 88x88 is cut out and enlarged to 176x176.
  6. Pixels are normalised with the training mean and std.
  7. Each model's raw outputs are divided by its temperature (calibration) and
     turned into probabilities; the models' probabilities are averaged.

Limits: the input must be a single cropped cell, not a whole microscope field.
The model knows three families only; any other pattern (or a negative cell)
still receives one of the three names. Research and demonstration use only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "resnet18_adjusted"
CLASS_NAMES = ["Homogeneous", "Speckled", "Nucleolar"]      # model output order
CANVAS = 100                    # training canvas size
MIN_CONFIDENCE = 0.70           # below this the answer is flagged "uncertain"


def to_grey(image):
    """Step 1: any PIL image -> 2-D uint8 array."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    pixels = np.asarray(image)
    if pixels.ndim == 3:                                    # colour: keep the brightest channel
        pixels = pixels[..., int(pixels.reshape(-1, pixels.shape[-1]).sum(axis=0).argmax())]
    if pixels.dtype != np.uint8:                            # 16-bit microscope files: rescale to 0..255
        pixels = pixels.astype(np.float32)
        pixels = (255 * pixels / max(float(pixels.max()), 1.0)).astype(np.uint8)
    return pixels


def to_canvas(grey):
    """Steps 2-4: shrink if needed, centre on the black canvas, stretch the brightness."""
    h, w = grey.shape
    if max(h, w) > CANVAS:
        scale = CANVAS / max(h, w)
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
        grey = np.asarray(Image.fromarray(grey).resize((w, h), Image.BILINEAR))
    cell = grey.astype(np.float32)
    low, high = np.percentile(cell, [5, 99.5])              # background level, bright-spot level
    high = max(high, low + 8)                               # very flat cells: do not blow up pure noise
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    top, left = (CANVAS - h) // 2, (CANVAS - w) // 2
    canvas[top:top + h, left:left + w] = np.clip((cell - low) / (high - low), 0, 1) * 255
    return canvas


class PatternPredictor:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, rounds=None, device=None):
        """rounds=None uses every trained round found (their probabilities are averaged)."""
        model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        temperatures = json.loads((model_dir / "calibration" / "temperatures.json").read_text())["temperature"]
        if rounds is None:
            rounds = sorted(int(p.parent.name.split("_")[1]) for p in model_dir.glob("round_*/best.pt"))
        assert rounds, f"no trained weights (round_*/best.pt) found in {model_dir}"
        self.members = []
        for k in rounds:
            info = json.loads((model_dir / f"round_{k}" / "metrics.json").read_text())
            assert info["mode"] == "adjusted", "this script applies the 'adjusted' brightness stretch"
            model = timm.create_model(info["model"], pretrained=False, num_classes=len(CLASS_NAMES), in_chans=1)
            model.load_state_dict(torch.load(model_dir / f"round_{k}" / "best.pt", map_location="cpu"))
            self.members.append({"round": k, "model": model.to(self.device).eval(), "mean": info["mean"],
                                 "std": info["std"], "temperature": temperatures[str(k)]})
        self.crop, self.input_size = info["crop"], info["input_size"]

    def _network_input(self, canvases):
        """Step 5: centre crop and enlargement, the same sampling the training notebook used."""
        x = torch.from_numpy(np.stack(canvases)).to(self.device).float().div(255).unsqueeze(1)
        scale = self.crop / CANVAS
        theta = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0]], device=self.device).repeat(len(x), 1, 1)
        grid = F.affine_grid(theta, (len(x), 1, self.input_size, self.input_size), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    @torch.no_grad()
    def logits(self, canvases):
        """Raw outputs per model: array of shape (models, images, classes)."""
        x = self._network_input(canvases)
        return np.stack([m["model"]((x - m["mean"]) / m["std"]).float().cpu().numpy() for m in self.members])

    def predict_canvases(self, canvases, min_confidence=MIN_CONFIDENCE):
        """Steps 6-7 for images that are already 100x100 stretched canvases."""
        temperatures = torch.tensor([m["temperature"] for m in self.members]).view(-1, 1, 1)
        probabilities = torch.softmax(torch.from_numpy(self.logits(canvases)) / temperatures, dim=2).mean(dim=0).numpy()
        results = []
        for p in probabilities:
            best = int(p.argmax())
            results.append({"family": CLASS_NAMES[best], "confidence": float(p[best]),
                            "uncertain": bool(p[best] < min_confidence),
                            "probabilities": {name: float(v) for name, v in zip(CLASS_NAMES, p)}})
        return results

    def predict(self, image, min_confidence=MIN_CONFIDENCE):
        """image: a file path or a PIL image of ONE cropped cell (colour or greyscale)."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        result = self.predict_canvases([to_canvas(to_grey(image))], min_confidence)[0]
        result["models_used"] = len(self.members)
        return result


def self_test(model_dir, n_per_round=40):
    """Run original dataset PNGs through the whole pipeline and compare with the outputs saved by
    the training notebook for the TEST fold of each round. Needs the dataset and image_logits.csv."""
    import pandas as pd
    cells_dir = PROJECT_ROOT.parent.parent / "datasets" / "cells" / "cells"
    rng = np.random.default_rng(42)
    worst, agree, total = 0.0, 0, 0
    for k in sorted(int(p.parent.name.split("_")[1]) for p in Path(model_dir).glob("round_*/best.pt")):
        saved = pd.read_csv(Path(model_dir) / f"round_{k}" / "image_logits.csv")
        saved = saved[saved["split"] == "test"].sample(n_per_round, random_state=int(rng.integers(1 << 30)))
        predictor = PatternPredictor(model_dir, rounds=[k])
        canvases = [to_canvas(to_grey(Image.open(cells_dir / f"{i}.png"))) for i in saved["file_index"]]
        new = predictor.logits(canvases)[0]
        old = saved[[f"logit_{c}" for c in CLASS_NAMES]].to_numpy()
        worst = max(worst, float(np.abs(new - old).max()))
        agree += int((new.argmax(1) == old.argmax(1)).sum()); total += len(saved)
        print(f"round {k}: largest difference in raw output {np.abs(new - old).max():.4f}, "
              f"same answer for {(new.argmax(1) == old.argmax(1)).sum()}/{len(saved)}")
    # Saved outputs came from a GPU in mixed precision, so tiny numeric differences are expected.
    assert worst < 0.1 and agree == total, "the script does NOT reproduce the saved predictions"
    print(f"SELF-TEST PASSED: {agree}/{total} answers identical, largest raw-output difference {worst:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Predict the ANA pattern family of cropped cell images.")
    parser.add_argument("images", nargs="*", help="image files, one cropped cell each")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder with round_k/ and calibration/")
    parser.add_argument("--rounds", type=int, nargs="+", help="use only these rounds (default: all found)")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--self-test", action="store_true", help="compare with the saved test predictions")
    args = parser.parse_args()
    if args.self_test:
        return self_test(args.model_dir)
    if not args.images:
        parser.error("give at least one image file, or --self-test")

    predictor = PatternPredictor(args.model_dir, args.rounds)
    results = {path: predictor.predict(path, args.min_confidence) for path in args.images}
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for path, r in results.items():
        shares = "  ".join(f"{name} {100 * p:.1f}%" for name, p in r["probabilities"].items())
        flag = "  [UNCERTAIN: below the confidence cut-off]" if r["uncertain"] else ""
        print(f"{path}\n  -> {r['family']} ({100 * r['confidence']:.1f}%){flag}\n     {shares}")
    print("Research and demonstration use only, not for diagnosis.")


if __name__ == "__main__":
    main()
