# Phase Plan: 3-Family ANA Pattern Classifier (cell level)

Living plan for Stage 2. Written 8 October 2026. Update the status column and the decision log as
phases finish.

## 1. Goal

Train a model that looks at **one HEp-2 cell image** and says which of three pattern families it shows,
with a confidence:

| Family | ICAP codes covered | Dataset label | Images |
|---|---|---|---|
| Homogeneous | AC-1 | 1 | 14,367 |
| Speckled | AC-4, AC-5 (merged) | 2 | 14,655 |
| Nucleolar | AC-8, AC-9, AC-10 (merged) | 3 | 13,257 |
| **Total** | | | **42,279** |

Centromere, nuclear membrane and Golgi (labels 4, 5, 6) are left out for now. The pipeline is written
so that adding a class later means changing one line in the config and re-running.

## 2. Dataset

- **What:** Qi et al. large-scale HEp-2 cell set, 63,445 single-cell images cut from the I3A Task-2
  specimen images. Paper: Qi, Zhao, Chen, Pietikäinen, *Pattern Recognition* 60 (2016) 420–429.
- **Where:** `..\..\datasets\cells\` → `cells\<n>.png`, `labels.mat`, `cells2.txt`, `cells.zip`.
- **Checked on 8 October 2026:** all files open, no exact duplicates, greyscale, sizes 62 to 100 px
  (typically 71×71), brightness untouched (no image reaches 255).
- **Known gaps:**
  - No specimen (patient slide) IDs.
  - Label numbers have no documented names; names were read from sample images.
  - No licence stated. Treat like AIDA: **never commit images or labels to GitHub**.

## 3. Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Classes | Homogeneous, Speckled, Nucleolar | Named in the proposal as the common patterns at Indus; balanced; plenty of data |
| Level | Cell images, one label each | That is what this dataset provides |
| Main model | ResNet-18, ImageNet-pretrained | Likely a little more accurate and more robust on new data than training from scratch; small, fast and well understood |
| Comparison | Postponed | Other pretrained models or a from-scratch run can be added later on the same folds |
| Framework | PyTorch on Colab T4; CPU steps on the laptop | Same as Stage 1 |
| Brightness | Never stretched or equalised | Project hard rule: brightness is signal |
| Evaluation | 5-fold cross-validation, mean ± std | A single split depends too much on luck |
| Headline metric | Balanced accuracy, plus per-class recall, F1, Cohen's kappa, confusion matrix | Matches the proposal's evaluation section |

Terms: a **CNN** (convolutional neural network) is the standard kind of network for images. **From
scratch** means the network starts with random numbers and learns only from our data. **Pretrained**
means it starts from a network already trained on ImageNet, a large set of everyday photos.
**Cross-validation** means splitting the data into 5 parts and training 5 times, each time testing on a
different part. **Balanced accuracy** is the average of the per-class accuracies, so each class counts
equally.

## 4. The main risk: no specimen IDs

Cells from one slide look alike. If cells of the same slide are in both training and test, the test
score is too optimistic (**leakage**). Without IDs we cannot group by slide directly.

Plan for handling it:

1. **Phase 2 checks whether file order follows specimens.** If neighbouring file numbers are much more
   alike than far-apart ones, the files were saved slide by slide.
2. **If yes, Phase 4 splits by contiguous blocks of file numbers** instead of at random. Each class is
   cut into 5 consecutive chunks. Then at most the one slide sitting on each cut is shared between two
   folds, instead of every slide being shared.
3. **If no, we fall back to a random split** and state clearly in every result that it is optimistic.
4. **If real IDs arrive** (from the I3A organisers or the paper's authors), Phase 4 is redone with a
   proper specimen-level split and training is re-run. Everything after Phase 4 is scripted so this is
   cheap.

Every reported number carries a note saying which of these splits produced it.

## 5. Phases

Each phase ends with a check. The next phase starts only after the check passes and the result has
been explained and agreed.

| # | Phase | Runs on | Status |
|---|---|---|---|
| 0 | Setup | Laptop + Colab | Laptop part done 8 Oct 2026; Colab check to be run by the user |
| 1 | Manifest and cleaning | Laptop | Not started |
| 2 | Exploratory data analysis (EDA) | Laptop | Not started |
| 3 | Preprocessing and packing | Laptop | Not started |
| 4 | Splitting into folds | Laptop | Not started |
| 5 | Baseline: pretrained ResNet-18 | Colab | Not started |
| 6 | Comparison with other models | Colab | Postponed (optional) |
| 7 | Calibration | Colab | Not started |
| 8 | Final evaluation and error analysis | Colab + laptop | Not started |
| 9 | Prediction script | Laptop | Not started |
| 10 | Write-up | Laptop | Not started |

### Phase 0: Setup

**Goal:** a clean project skeleton so every later step has a fixed place.

1. Create `scripts/`, `notebooks/`, `outputs/`, `data/`, `Docs/`.
2. Write `scripts/config.py`: dataset path, the class list (`{1: Homogeneous, 2: Speckled, 3: Nucleolar}`),
   image size, seed 42, number of folds.
3. Write `.gitignore` that blocks `data/`, all images, `.mat`, `.zip`, `.npz` and model weights.
4. Write `requirements-local.txt`.
5. Rewrite `CLAUDE.md` for this stage (the current one describes Stage 1).
6. Create the Drive folder `My Drive/ANA_Pattern3/` and a Colab check notebook (GPU visible, Drive mounts).

**Check:** config imports without error; `.gitignore` test shows no data file would be committed; Colab
sees the T4.

### Phase 1: Manifest and cleaning

**Goal:** one table that lists every image we will use. A **manifest** is that table; all later steps
read it instead of touching the raw folder.

1. Read `labels.mat`; pair entry *i* with `cells/i.png`.
2. Keep labels 1, 2, 3. Write `data/manifest.csv` with columns `image_path`, `sample_id`, `label`,
   `label_name`, `source`, `file_index`. `sample_id` stays empty until Phase 4 because no IDs exist.
3. Safety checks in code: every listed file exists and opens; counts equal 14,367 / 14,655 / 13,257;
   no duplicate rows; no exact-duplicate image content.
4. Write a short cleaning log to `outputs/`.

**Check:** 42,279 rows, three labels, zero failures in the log.

### Phase 2: EDA

**Goal:** understand the images and hunt for shortcuts before training. A **shortcut** is something
unrelated to the pattern that still reveals the label, such as image size.

1. Per class: width, height and aspect-ratio distributions.
2. Per class: mean, spread and maximum of brightness.
3. **Shortcut tests:** how well can the class be guessed from size alone, and from brightness alone?
   (Stage 1 found a size shortcut this way.)
4. **File-order test** (section 4): similarity of neighbouring files versus random pairs, and a plot of
   brightness against file number to see whether slide boundaries are visible.
5. Sample sheets per class and a look at the darkest, brightest and smallest images for junk.
6. Charts and a summary table in `outputs/eda/`.

**Check:** a written answer to three questions: is there a size shortcut, is there a brightness
shortcut, does file order follow specimens.

### Phase 3: Preprocessing and packing

**Goal:** turn 42,279 different-sized images into one fixed-size array Colab can load in seconds.

1. Keep greyscale (one channel). Pixel values are not changed.
2. Place each image in the centre of a 100×100 black canvas (**padding**). No resizing, so nothing is
   blurred and every cell keeps its true scale. 100 is the largest size in the dataset, so nothing is cut.
3. Save all images as one `uint8` array plus labels and file indices in `data/cells3_100.npz`
   (about 420 MB in memory, far less on disk).
4. Verify by reloading: shape, counts, and that a few random images are pixel-identical to the originals
   inside the padded area.
5. Upload the `.npz` and the manifest to Drive.

If Phase 2 finds a size shortcut, padding is replaced by a fixed resize before this phase is run.

**Check:** reload test passes; file is on Drive.

### Phase 4: Splitting

**Goal:** fix once which images are used for training, validation and testing in each of the 5 rounds.

1. Use the Phase 2 answer to choose contiguous-block or random splitting (section 4).
2. Assign a `fold` (0–4) to every row, stratified by class so each fold has the same class mix.
3. Per round *k*: test = fold *k*, validation = fold *(k+1) mod 5*, training = the other three.
   **Validation** is the part used to pick the best epoch; **test** is touched only for the final score.
4. Code checks: every image is in exactly one fold; class proportions per fold; no block in two folds.
5. Save the fold column into the manifest and re-upload.

**Check:** fold table printed and agreed; all assertions pass.

### Phase 5: Baseline, pretrained ResNet-18

**Goal:** the first real result.

- **Model:** `timm` ResNet-18 with ImageNet weights, `num_classes=3`, `in_chans=1` (timm converts the
  pretrained colour filters of the first layer to one greyscale channel). About 11 million weights.
  All layers are fine-tuned.
- **Input:** random 96×96 crop in training, centre 96×96 crop in validation and test.
- **Augmentation** (random changes that create variety): horizontal and vertical flips, rotation by any
  angle, brightness ±10% (same limit as Stage 1). Rotation matters because a cell has no "up".
- **Normalisation:** subtract the mean and divide by the standard deviation of the training pixels.
  This is one fixed formula for all images, not a per-image stretch.
- **Loss:** cross-entropy over 3 classes (the standard loss when exactly one class is correct).
- **Training:** AdamW, learning rate 3e-4, weight decay 1e-2, 1 warm-up epoch then cosine decay, mixed
  precision, batch 128, at most 30 epochs, early stopping with patience 7 on validation balanced
  accuracy. These are the Stage 1 defaults and are not tuned.
- **Saved per round to Drive:** best weights, the training log, and the raw outputs (logits) for every
  validation and test image. Rounds are resumable.
- **One extra diagnostic run:** the same model on a plain random split, to measure how much leakage
  inflates the score.

**Check:** 5 rounds finished; table of balanced accuracy, per-class recall, F1 and kappa as mean ± std;
confusion matrix; learning curves look sane (no collapse, no wild overfitting).

### Phase 6: Comparison (postponed, optional)

**Goal:** check whether another model does better than the baseline.

1. Candidates, on the same folds with the same recipe: EfficientNet-B0 pretrained (the Stage 1 model),
   and a from-scratch run to measure what pretraining adds.
2. One table with all models; choice made on **validation** scores, with size and speed as
   tie-breakers.

**Check:** comparison table and a one-paragraph decision with its reason.

### Phase 7: Calibration

**Goal:** make the confidence honest. **Calibration** means that when the model says 90%, it is right
about 90% of the time.

1. Fit temperature scaling (one number that softens or sharpens the confidences) on validation logits.
2. Report the calibration error before and after, with a reliability chart.

**Check:** calibration error does not get worse; temperature recorded.

### Phase 8: Final evaluation and error analysis

**Goal:** the numbers that go in the report, and an understanding of the mistakes.

1. Final test-fold metrics for the chosen model.
2. Error study: which pairs are confused (expect Homogeneous ↔ Speckled), image sheets of the most
   confident mistakes, accuracy against cell brightness and cell size.
3. Context only, not a like-for-like comparison: the paper's best hand-crafted method reached 86.61%
   mean class accuracy on all six classes.

**Check:** results table, confusion matrix and a short list of failure types.

### Phase 9: Prediction script

**Goal:** something the website project can call.

1. `predict_pattern.py`: one cell image in → family, confidence, all three probabilities out, using the
   exact preprocessing from Phase 3 and the temperature from Phase 7.
2. A written inference recipe for the website team.
3. State the limit plainly: input must be a single cropped cell, not a whole microscope field.

**Check:** script reproduces the saved test predictions for a sample of images.

### Phase 10: Write-up

Update this plan and `CLAUDE.md`, and produce the explanation document for the supervisor.

## 6. Folder layout

```
Model Training on Basic Classes/
  CLAUDE.md
  Docs/        plans and explanations
  scripts/     config.py, build_manifest.py, eda.py, preprocess.py, make_folds.py, predict_pattern.py
  notebooks/   00_colab_setup, 01_train_baseline, 02_compare_models, 03_calibrate, 04_evaluate
  outputs/     logs, tables, charts (small files only)
  data/        manifest.csv, cells3_100.npz   (git-ignored)
```

## 7. Open items

- Specimen IDs: waiting on the I3A organisers; consider emailing the paper's authors too.
- Confirm the label-to-name mapping from the paper once a copy is obtained.
- How cell-level predictions will be used on whole-field Indus images (needs a cell-cropping step and
  the image-to-sample aggregation from the proposal). Out of scope for this plan.
- Adding Centromere (AC-3) as a fourth class after the three-class model works.
- The Task-1 copy (13,596 cells) as a second check. Not an independent test: same laboratory, possible
  shared specimens, and that copy has altered contrast.

## 8. Decision log

| Date | Decision |
|---|---|
| 8 Oct 2026 | Use the Qi 63,445-cell dataset; set I3A Task-1 aside until the official copy arrives |
| 8 Oct 2026 | Three families: Homogeneous, Speckled, Nucleolar. Centromere postponed |
| 8 Oct 2026 | Pretrained only for now (ResNet-18). From-scratch and other models moved to an optional Phase 6 |
| 8 Oct 2026 | Own public GitHub repository `ANA-Families-Classification` (code and docs only; dataset files stay ignored) |
