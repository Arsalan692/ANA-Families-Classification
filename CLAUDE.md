# CLAUDE.md

Context for Claude Code sessions on this project. Last updated 8 October 2026.

## Who and what

- **User:** Arsalan Ahmed, CS undergraduate at FAST-NUCES. FYP supervisor: Ubaid Chawla.
- **FYP:** *Automated Classification of ANA Immunofluorescence Patterns Using Deep Learning*.
  A lab photographs HEp-2 cells under a fluorescence microscope (the ANA IIF test). The system should
  first say whether an image is ANA-Positive or ANA-Negative, then name the staining pattern.
- **This folder** holds **Stage 2: a cell-level pattern classifier for three families**,
  Homogeneous (AC-1), Speckled (AC-4/5) and Nucleolar (AC-8/9/10), using a pretrained ResNet-18.
- **Stage 1** (image-level Positive/Negative classifier on AIDA, EfficientNet-B0, finished up to
  calibration) lives in `../../Second Model Training`. Its `CLAUDE.md` and
  `Docs/Phase_Plan_ANA_PosNeg.md` hold the details. It is background and a source of reusable method.
- **The plan for this stage:** `Docs/Phase_Plan_Pattern_3Class.md`. **Read it first.** It holds the
  phases, their status, the decision log and the open items.

## How the user wants to work

- Explain everything in simple language and define every technical term the first time it appears.
- Give the *why* for every decision, not only the *what*.
- Work **phase by phase**. Finish and check one phase before starting the next, and explain each
  phase when it is completed.
- Answer concisely when asked to; the user is often short on time.
- The FYP proposal is an early draft, not a roadmap. Recommend on technical merit.
- Training runs on **Google Colab (T4 GPU)** with **PyTorch**. CPU-only steps (cleaning, EDA,
  preprocessing, splitting) run on the Windows laptop. Results are saved to a private Google Drive.
- The user runs the notebooks in Colab and copies outputs back into `outputs/` for checking.

## Hard rules

- **No dataset files in any public place.** The Qi cell dataset has no stated licence and is derived
  from the access-controlled I3A data. No images, `labels.mat`, manifest or packed arrays on GitHub.
  Private Google Drive only. `.gitignore` enforces this; keep it that way. The same applies to AIDA.
- **Pixels only.** No metadata is given to the model.
- **Specimen / file-order information is used only for splitting**, never given to the model.
- **Do not change image brightness or contrast** (no contrast stretching, histogram equalisation,
  auto-levels). In ANA images brightness is the signal. A single fixed mean/std normalisation applied
  identically to every image is allowed.
- **Test data is used once, at the end.** Epoch selection, early stopping and model choice use
  validation only.
- Results are for research and demonstration, not diagnosis.
- Git commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## The dataset

- **Qi et al. large-scale HEp-2 cell set**: 63,445 single-cell greyscale PNGs cut from the I3A Task-2
  specimen images. Paper: Qi, Zhao, Chen, Pietikäinen, *Pattern Recognition* 60 (2016) 420–429
  (paywalled, not yet obtained). Download page: `https://qixianbiao.github.io/HEp2Cell/`.
- **Location:** `../../datasets/cells/` → `cells/<n>.png`, `labels.mat` (entry *i* is the label of
  `i.png`), `cells2.txt` (file list only), `cells.zip`. Read-only.
- **Labels** (numbers only in the file; names read from sample images, not documented):

  | Label | Family | AC codes | Images | Used now |
  |---|---|---|---|---|
  | 1 | Homogeneous | AC-1 | 14,367 | Yes |
  | 2 | Speckled | AC-4, AC-5 | 14,655 | Yes |
  | 3 | Nucleolar | AC-8, AC-9, AC-10 | 13,257 | Yes |
  | 4 | Centromere | AC-3 | 13,737 | Later |
  | 5 | Nuclear membrane | AC-11, AC-12 | 5,086 | No |
  | 6 | Golgi | AC-22 | 2,343 | No |

- **Checked 8 October 2026:** all files open, no exact duplicates, sizes 62 to 100 px (typically
  71×71), brightness untouched (no image reaches 255).
- **No specimen IDs.** Cells from one slide look alike, so a random split leaks. The plan (section 4)
  tests whether file order follows specimens and, if so, splits by contiguous blocks of file numbers.
  Every reported number must say which kind of split produced it.
- **Published reference on all six classes** (hand-crafted descriptors, split protocol unknown):
  86.61% mean class accuracy; Homogeneous ↔ Speckled is the most confused pair.

## Other datasets on disk

- `../../datasets/I3A Task-1/`: 13,596 cells in six class folders. An unofficial copy: every image is
  78×78 and contrast appears stretched, and it has no specimen IDs. Set aside. The official copy
  (with specimen IDs) was requested by email from `hep2benchmarking@gmail.com` on 8 October 2026.
- AIDA (`../../Model Training/aida_project_database`): unsuitable for single-pattern work. Pure
  AC-1 / AC-2 / AC-3 images number only 21 / 21 / 26 (7 / 7 / 8 patients).

## Where things are

| Path | Contents |
|---|---|
| `Docs/Phase_Plan_Pattern_3Class.md` | The living plan |
| `scripts/config.py` | All paths and settings (class list, sizes, seed, folds) |
| `scripts/` | Laptop pipeline, added phase by phase |
| `notebooks/` | Colab notebooks, starting with `00_colab_setup` |
| `outputs/` | Logs, metric tables, charts. Anything embedding dataset images goes in a `samples/` subfolder (git-ignored) |
| `data/` | Manifest and packed image array. **Git-ignored.** |
| Google Drive | `My Drive/ANA_Pattern3/data/` and `My Drive/ANA_Pattern3/outputs/` |
| GitHub | `https://github.com/Arsalan692/ANA-Families-Classification.git`, branch `main`, public (code and docs only) |

## Method carried over from Stage 1

- Dataset-independent manifest: `image_path`, `sample_id`, `label`, `source`, plus dataset extras.
- Clean before training, log every change, add safety checks in code.
- Look for shortcuts in the EDA (class versus image size, class versus brightness).
- Cross-validate with 5 folds and report mean ± std; report balanced accuracy and per-class recall.
- Save raw logits per image so later analysis needs no retraining. Make rounds resumable.
- Copy one file to Colab's local disk and hold images in memory.
- Training defaults: AdamW, lr 3e-4, weight decay 1e-2, 1 warm-up epoch then cosine decay, mixed
  precision, early stopping on validation balanced accuracy. Not tuned.
